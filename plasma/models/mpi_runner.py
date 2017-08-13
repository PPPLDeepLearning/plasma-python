'''
#########################################################
This file trains a deep learning model to predict
disruptions on time series data from plasma discharges.

Dependencies:
conf.py: configuration of model,training,paths, and data
model_builder.py: logic to construct the ML architecture
data_processing.py: classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
import os
import sys 
import time
import datetime
import numpy as np

from functools import partial
import socket
sys.setrecursionlimit(10000)
import getpass

#import keras sequentially because it otherwise reads from ~/.keras/keras.json with too many threads.
#from mpi_launch_tensorflow import get_mpi_task_index 
import mpi4py
from mpi4py import MPI
comm = MPI.COMM_WORLD
task_index = comm.Get_rank()
num_workers = comm.Get_size()

from pprint import pprint
from plasma.conf import conf
from plasma.utils.state_reset import reset_states,get_states
from plasma.models.loader import ProcessGenerator

NUM_GPUS = conf['num_gpus']
MY_GPU = task_index % NUM_GPUS

backend = conf['model']['backend']

if backend == 'tf' or backend == 'tensorflow':
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(MY_GPU)#,mode=NanGuardMode'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    set_session(tf.Session(config=config))
else:
    os.environ['KERAS_BACKEND'] = 'theano'
    base_compile_dir = '{}/tmp/{}-{}'.format(conf['paths']['output_path'],socket.gethostname(),task_index)
    os.environ['THEANO_FLAGS'] = 'device=gpu{},floatX=float32,base_compiledir={}'.format(MY_GPU,base_compile_dir)#,mode=NanGuardMode'
    import theano
#import keras
for i in range(num_workers):
  comm.Barrier()
  if i == task_index:
    print('[{}] importing Keras'.format(task_index))
    from keras import backend as K
    from keras.layers import Input,Dense, Dropout
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Model
    from keras.optimizers import *
    from keras.utils.generic_utils import Progbar 
    import keras.callbacks as cbks

from plasma.models import builder
from plasma.utils.evaluation import get_loss_from_list
from plasma.utils.processing import concatenate_sublists
from plasma.utils.performance import PerformanceAnalyzer
from plasma.primitives.ops import mpi_sum_f16

if task_index == 0:
    pprint(conf)



class MPIOptimizer(object):
  def __init__(self,lr):
    self.lr = lr
    self.iterations = 0

  def get_deltas(self,raw_deltas):
    raise NotImplementedError

  def set_lr(self,lr):
    self.lr = lr

class MPISGD(MPIOptimizer):
  def __init__(self,lr):
    super(MPISGD,self).__init__(lr)

  def get_deltas(self,raw_deltas):
    deltas = []
    for g in raw_deltas:
      deltas.append(self.lr*g)

    self.iterations += 1
    return deltas

class MPIMomentumSGD(MPIOptimizer):
    def __init__(self, lr):
        super(MPIMomentumSGD, self).__init__(lr)
        self.momentum = 0.9

    def get_deltas(self, raw_deltas): 
        deltas = []

        if self.iterations == 0:
            self.velocity_list = [np.zeros_like(g) for g in raw_deltas]

        for (i,g) in enumerate(raw_deltas):
            self.velocity_list[i] = self.momentum * self.velocity_list[i] + self.lr * g
            deltas.append(self.velocity_list[i])

        self.iterations += 1

        return deltas

class MPIAdam(MPIOptimizer):
  def __init__(self,lr):
    super(MPIAdam,self).__init__(lr)
    self.beta_1 = 0.9
    self.beta_2 = 0.999
    self.eps = 1e-8

  def get_deltas(self,raw_deltas):

    if K.floatx() == "float16":
        raw_deltas[:] = map(lambda w: w.astype(np.float32),raw_deltas)

    if self.iterations == 0:
      self.m_list = [np.zeros_like(g) for g in raw_deltas]
      self.v_list = [np.zeros_like(g) for g in raw_deltas]

    t = self.iterations + 1
    lr_t = self.lr * np.sqrt(1-self.beta_2**t)/(1-self.beta_1**t)
    deltas = []
    for (i,g) in enumerate(raw_deltas):
      m_t = (self.beta_1 * self.m_list[i]) + (1 - self.beta_1) * g
      v_t = (self.beta_2 * self.v_list[i]) + (1 - self.beta_2) * (g**2)
      delta_t = lr_t * m_t / (np.sqrt(v_t) + self.eps)
      deltas.append(delta_t)
      self.m_list[i] = m_t
      self.v_list[i] = v_t

    self.iterations += 1

    if K.floatx() == "float16":
        deltas[:] = map(lambda w: w.astype(np.float16),deltas)

    return deltas


class Averager(object):
  def __init__(self):
    self.steps = 0
    self.val = 0.0

  def add_val(self,val):
    self.val = (self.steps * self.val + 1.0 * val)/(self.steps + 1.0)
    self.steps += 1

  def get_val(self):
    return self.val


class MPIModel():
  def __init__(self,model,optimizer,comm,batch_iterator,batch_size,num_replicas=None,warmup_steps=1000,lr=0.01,num_batches_minimum=100):
    # random.seed(task_index)
    self.epoch = 0
    self.num_so_far = 0
    self.num_so_far_accum = 0
    self.num_so_far_indiv = 0
    self.model = model
    self.optimizer = optimizer
    self.max_lr = 0.1
    self.lr = lr if (lr < self.max_lr) else self.max_lr
    self.DUMMY_LR = 0.001
    self.comm = comm
    self.batch_size = batch_size
    self.batch_iterator = batch_iterator
    self.set_batch_iterator_func()
    self.warmup_steps=warmup_steps
    self.num_batches_minimum=num_batches_minimum
    self.num_workers = comm.Get_size()
    self.task_index = comm.Get_rank()
    self.history = cbks.History()
    if num_replicas is None or num_replicas < 1 or num_replicas > self.num_workers:
        self.num_replicas = self.num_workers
    else:
        self.num_replicas = num_replicas


  def set_batch_iterator_func(self):
    self.batch_iterator_func = ProcessGenerator(self.batch_iterator())

  def close(self):
    self.batch_iterator_func.__exit__()

  def set_lr(self,lr):
    self.lr = lr

  def save_weights(self,path,overwrite=False):
    self.model.save_weights(path,overwrite=overwrite)

  def load_weights(self,path):
    self.model.load_weights(path)

  def compile(self,optimizer,loss='mse'):
    if optimizer == 'sgd':
        optimizer_class = SGD
    elif optimizer == 'adam':
        optimizer_class = Adam
    elif optimizer == 'rmsprop':
        optimizer_class = RMSprop
    elif optimizer == 'nadam':
        optimizer_class = Nadam
    else:
        print("Optimizer not implemented yet")
        exit(1)
    self.model.compile(optimizer=optimizer_class(lr=self.DUMMY_LR),loss=loss)



  def train_on_batch_and_get_deltas(self,X_batch,Y_batch,verbose=False):
    '''
    The purpose of the method is to perform a single gradient update over one mini-batch for one model replica.
    Given a mini-batch, it first accesses the current model weights, performs single gradient update over one mini-batch,
    gets new model weights, calculates weight updates (deltas) by subtracting weight scalars, applies the learning rate.

    It performs calls to: subtract_params, multiply_params 

    Argument list: 
      - X_batch: input data for one mini-batch as a Numpy array
      - Y_batch: labels for one mini-batch as a Numpy array
      - verbose: set verbosity level (currently unused)

    Returns:  
      - deltas: a list of model weight updates
      - loss: scalar training loss
    '''
    weights_before_update = self.model.get_weights()

    loss = self.model.train_on_batch(X_batch,Y_batch)

    weights_after_update = self.model.get_weights()
    self.model.set_weights(weights_before_update)

    deltas = subtract_params(weights_after_update,weights_before_update)
    deltas = multiply_params(deltas,1.0/self.DUMMY_LR)

    return deltas,loss


  def get_new_weights(self,deltas):
    return add_params(self.model.get_weights(),deltas)

  def mpi_average_gradients(self,arr,num_replicas=None):
    if num_replicas == None:
      num_replicas = self.num_workers 
    if self.task_index >= num_replicas:
      arr *= 0.0
    arr_global = np.empty_like(arr)
    if K.floatx() == 'float16':
        self.comm.Allreduce(arr,arr_global,op=mpi_sum_f16)
    else:
        self.comm.Allreduce(arr,arr_global,op=MPI.SUM)
    arr_global /= num_replicas
    return arr_global



  def mpi_average_scalars(self,val,num_replicas=None):
    '''
    The purpose of the method is to calculate a simple scalar arithmetic mean over num_replicas.

    It performs calls to: MPIModel.mpi_sum_scalars

    Argument list: 
      - val: value averaged, scalar
      - num_replicas: the size of the ensemble an average is perfromed over

    Returns:  
      - val_global: scalar arithmetic mean over num_replicas
    '''
    val_global = self.mpi_sum_scalars(val,num_replicas)
    val_global /= num_replicas
    return val_global


  def mpi_sum_scalars(self,val,num_replicas=None):
    '''
    The purpose of the method is to calculate a simple scalar arithmetic mean over num_replicas using MPI allreduce action with fixed op=MPI.SIM

    Argument list: 
      - val: value averaged, scalar
      - num_replicas: the size of the ensemble an average is perfromed over

    Returns:  
      - val_global: scalar arithmetic mean over num_replicas
    '''
    if num_replicas == None:
      num_replicas = self.num_workers 
    if self.task_index >= num_replicas:
      val *= 0.0
    val_global = 0.0 
    val_global = self.comm.allreduce(val,op=MPI.SUM)
    return val_global



  def sync_deltas(self,deltas,num_replicas=None):
    global_deltas = []
    #default is to reduce the deltas from all workers
    for delta in deltas:
      global_deltas.append(self.mpi_average_gradients(delta,num_replicas))
    return global_deltas 

  def set_new_weights(self,deltas,num_replicas=None):
    global_deltas = self.sync_deltas(deltas,num_replicas)
    effective_lr = self.get_effective_lr(num_replicas)

    self.optimizer.set_lr(effective_lr)
    global_deltas = self.optimizer.get_deltas(global_deltas)

    if self.comm.rank == 0:
      new_weights = self.get_new_weights(global_deltas)
    else:
      new_weights = None
    new_weights = self.comm.bcast(new_weights,root=0)
    self.model.set_weights(new_weights)

  def build_callbacks(self,conf,callbacks_list):
      '''
      The purpose of the method is to set up logging and history. It is based on Keras Callbacks
      https://github.com/fchollet/keras/blob/fbc9a18f0abc5784607cd4a2a3886558efa3f794/keras/callbacks.py

      Currently used callbacks include: BaseLogger, CSVLogger, EarlyStopping. 
      Other possible callbacks to add in future: RemoteMonitor, LearningRateScheduler

      Argument list: 
        - conf: There is a "callbacks" section in conf.yaml file. Relevant parameters are:
             list: Parameter specifying additional callbacks, read in the driver script and passed as an argument of type list (see next arg)
             metrics: List of quantities monitored during training and validation
             mode: one of {auto, min, max}. The decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity. 
             monitor: Quantity used for early stopping, has to be from the list of metrics
             patience: Number of epochs used to decide on whether to apply early stopping or continue training
        - callbacks_list: uses callbacks.list configuration parameter, specifies the list of additional callbacks
      Returns: modified list of callbacks
      '''

      mode = conf['callbacks']['mode']
      monitor = conf['callbacks']['monitor']
      patience = conf['callbacks']['patience']
      csvlog_save_path = conf['paths']['csvlog_save_path']
      #CSV callback is on by default
      if not os.path.exists(csvlog_save_path):
          os.makedirs(csvlog_save_path)

      callbacks_list = conf['callbacks']['list']

      callbacks = [cbks.BaseLogger()]
      callbacks += [self.history]
      callbacks += [cbks.CSVLogger("{}callbacks-{}.log".format(csvlog_save_path,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))]

      if "earlystop" in callbacks_list: 
          callbacks += [cbks.EarlyStopping(patience=patience, monitor=monitor, mode=mode)]
      if "lr_scheduler" in callbacks_list: 
          pass
      
      return cbks.CallbackList(callbacks)

  def train_epoch(self):
    '''
    The purpose of the method is to perform distributed mini-batch SGD for one epoch.
    It takes the batch iterator function and a NN model from MPIModel object, fetches mini-batches
    in a while-loop until number of samples seen by the ensemble of workers (num_so_far) exceeds the 
    training dataset size (num_total). 

    During each iteration, the gradient updates (deltas) and the loss are calculated for each model replica
    in the ensemble, weights are averaged over ensemble, and the new weights are set.

    It performs calls to: MPIModel.get_deltas, MPIModel.set_new_weights methods 

    Argument list: Empty

    Returns:  
      - step: epoch number
      - ave_loss: training loss averaged over replicas
      - curr_loss:
      - num_so_far: the number of samples seen by ensemble of replicas to a current epoch (step) 

    Intermediate outputs and logging: debug printout of task_index (MPI), epoch number, number of samples seen to 
    a current epoch, average training loss
    '''

    verbose = False
    first_run = True
    step = 0
    loss_averager = Averager()
    t_start = time.time()

    batch_iterator_func = self.batch_iterator_func
    num_total = 1
    ave_loss = -1
    curr_loss = -1
    t0 = 0 
    t1 = 0 
    t2 = 0

    while (self.num_so_far-self.epoch*num_total) < num_total or step < self.num_batches_minimum:

      try:
          batch_xs,batch_ys,batches_to_reset,num_so_far_curr,num_total,is_warmup_period = next(batch_iterator_func)
      except StopIteration:
          print("Resetting batch iterator.")
          self.num_so_far_accum = self.num_so_far_indiv
          self.set_batch_iterator_func()
          batch_iterator_func = self.batch_iterator_func
          batch_xs,batch_ys,batches_to_reset,num_so_far_curr,num_total,is_warmup_period = next(batch_iterator_func)
      self.num_so_far_indiv = self.num_so_far_accum+num_so_far_curr

      # if batches_to_reset:
        # self.model.reset_states(batches_to_reset)

      warmup_phase = (step < self.warmup_steps and self.epoch == 0)
      num_replicas = 1 if warmup_phase else self.num_replicas

      self.num_so_far = self.mpi_sum_scalars(self.num_so_far_indiv,num_replicas)

      #run the model once to force compilation. Don't actually use these values.
      if first_run:
        first_run = False
        t0_comp = time.time()
        _,_ = self.train_on_batch_and_get_deltas(batch_xs,batch_ys,verbose)
        self.comm.Barrier()
        sys.stdout.flush()
        print_unique('Compilation finished in {:.2f}s'.format(time.time()-t0_comp))
        t_start = time.time()
        sys.stdout.flush()  
      
      if np.any(batches_to_reset):
        reset_states(self.model,batches_to_reset)

      t0 = time.time()
      deltas,loss = self.train_on_batch_and_get_deltas(batch_xs,batch_ys,verbose)
      t1 = time.time()
      if not is_warmup_period:
        self.set_new_weights(deltas,num_replicas)
        t2 = time.time()
        write_str_0 = self.calculate_speed(t0,t1,t2,num_replicas)

        curr_loss = self.mpi_average_scalars(1.0*loss,num_replicas)
        #if self.task_index == 0:
          #print(self.model.get_weights()[0][0][:4])
        loss_averager.add_val(curr_loss)
        ave_loss = loss_averager.get_val()
        eta = self.estimate_remaining_time(t0 - t_start,self.num_so_far-self.epoch*num_total,num_total)
        write_str = '\r[{}] step: {} [ETA: {:.2f}s] [{:.2f}/{}], loss: {:.5f} [{:.5f}] | '.format(self.task_index,step,eta,1.0*self.num_so_far,num_total,ave_loss,curr_loss)
        print_unique(write_str + write_str_0)
        step += 1
      else:
        print_unique('\r[{}] warmup phase, num so far: {}'.format(self.task_index,self.num_so_far))
        

      

    effective_epochs = 1.0*self.num_so_far/num_total
    epoch_previous = self.epoch
    self.epoch = effective_epochs
    print_unique('\nEpoch {:.2f} finished ({:.2f} epochs passed) in {:.2f} seconds.\n'.format(1.0*self.epoch,self.epoch-epoch_previous,t2 - t_start))
    return (step,ave_loss,curr_loss,self.num_so_far,effective_epochs)


  def estimate_remaining_time(self,time_so_far,work_so_far,work_total):
    eps = 1e-6
    total_time = 1.0*time_so_far*work_total/(work_so_far + eps)
    return total_time - time_so_far

  def get_effective_lr(self,num_replicas):
    effective_lr = self.lr * num_replicas
    if effective_lr > self.max_lr:
      print_unique('Warning: effective learning rate set to {}, larger than maximum {}. Clipping.'.format(effective_lr,self.max_lr))
      effective_lr = self.max_lr
    return effective_lr

  def get_effective_batch_size(self,num_replicas):
    return self.batch_size*num_replicas

  def calculate_speed(self,t0,t_after_deltas,t_after_update,num_replicas,verbose=False):
    effective_batch_size = self.get_effective_batch_size(num_replicas)
    t_calculate = t_after_deltas - t0
    t_sync = t_after_update - t_after_deltas
    t_tot = t_after_update - t0

    examples_per_sec = effective_batch_size/t_tot
    frac_calculate = t_calculate/t_tot
    frac_sync = t_sync/t_tot

    print_str = '{:.2E} Examples/sec | {:.2E} sec/batch [{:.1%} calc., {:.1%} synch.]'.format(examples_per_sec,t_tot,frac_calculate,frac_sync)
    print_str += '[batch = {} = {}*{}] [lr = {:.2E} = {:.2E}*{}]'.format(effective_batch_size,self.batch_size,num_replicas,self.get_effective_lr(num_replicas),self.lr,num_replicas)
    if verbose:
      print_unique(print_str)
    return print_str



def print_unique(print_str):
  if task_index == 0:
    sys.stdout.write(print_str)
    sys.stdout.flush()

def print_all(print_str):
  sys.stdout.write('[{}] '.format(task_index) + print_str)
  sys.stdout.flush()


def multiply_params(params,eps):
  return [el*eps for el in params]

def subtract_params(params1,params2):
  return [p1 - p2 for p1,p2 in zip(params1,params2)]

def add_params(params1,params2):
  return [p1 + p2 for p1,p2 in zip(params1,params2)]


def get_shot_list_path(conf):
    return conf['paths']['base_path'] + '/normalization/shot_lists.npz' #kyle: not compatible with flexible conf.py hierarchy 

def save_shotlists(conf,shot_list_train,shot_list_validate,shot_list_test):
    path = get_shot_list_path(conf)
    np.savez(path,shot_list_train=shot_list_train,shot_list_validate=shot_list_validate,shot_list_test=shot_list_test)

def load_shotlists(conf):
    path = get_shot_list_path(conf)
    data = np.load(path)
    shot_list_train = data['shot_list_train'][()]
    shot_list_validate = data['shot_list_validate'][()]
    shot_list_test = data['shot_list_test'][()]
    return shot_list_train,shot_list_validate,shot_list_test

#shot_list_train,shot_list_validate,shot_list_test = load_shotlists(conf)

def mpi_make_predictions(conf,shot_list,loader,custom_path=None):
    shot_list.sort()#make sure all replicas have the same list
    specific_builder = builder.ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []

    model = specific_builder.build_model(True)
    specific_builder.load_model_weights(model,custom_path)

    #broadcast model weights then set it explicitely: fix for Py3.6
    if sys.version_info[0] > 2:
        if task_index == 0:
            new_weights = model.get_weights()
        else:
            new_weights = None
        nw = comm.bcast(new_weights,root=0)
        model.set_weights(nw)

    model.reset_states()
    if task_index == 0:
        pbar =  Progbar(len(shot_list))
    shot_sublists = shot_list.sublists(conf['model']['pred_batch_size'],do_shuffle=False,equal_size=True)

    y_prime_global = []
    y_gold_global = []
    disruptive_global = []
    if task_index != 0:
        loader.verbose = False

    for (i,shot_sublist) in enumerate(shot_sublists):

        if i % num_workers == task_index:
            X,y,shot_lengths,disr = loader.load_as_X_y_pred(shot_sublist)


        
            #load data and fit on data
            y_p = model.predict(X,batch_size=conf['model']['pred_batch_size'])
            model.reset_states()
            y_p = loader.batch_output_to_array(y_p)
            y = loader.batch_output_to_array(y)

            #cut arrays back
            y_p = [arr[:shot_lengths[j]] for (j,arr) in enumerate(y_p)]
            y = [arr[:shot_lengths[j]] for (j,arr) in enumerate(y)]

            # print('Shots {}/{}'.format(i*num_at_once + j*1.0*len(shot_sublist)/len(X_list),len(shot_list_train)))
            y_prime += y_p
            y_gold += y
            disruptive += disr
            # print_all('\nFinished with i = {}'.format(i))

        if i % num_workers == num_workers -1 or i == len(shot_sublists) - 1:
            comm.Barrier()
            y_prime_global += concatenate_sublists(comm.allgather(y_prime))
            y_gold_global += concatenate_sublists(comm.allgather(y_gold))
            disruptive_global += concatenate_sublists(comm.allgather(disruptive))
            comm.Barrier()
            y_prime = []
            y_gold = []
            disruptive = []
            # print_all('\nFinished subepoch with lists len(y_prime_global), gold, disruptive = {},{},{}'.format(len(y_prime_global),len(y_gold_global),len(disruptive_global)))

        if task_index == 0:
            pbar.add(1.0*len(shot_sublist))

    y_prime_global = y_prime_global[:len(shot_list)]
    y_gold_global = y_gold_global[:len(shot_list)]
    disruptive_global = disruptive_global[:len(shot_list)]

    return y_prime_global,y_gold_global,disruptive_global


def mpi_make_predictions_and_evaluate(conf,shot_list,loader,custom_path=None):
    y_prime,y_gold,disruptive = mpi_make_predictions(conf,shot_list,loader,custom_path)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    shot_list.set_weights(analyzer.get_shot_difficulty(y_prime,y_gold,disruptive))
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'])
    return y_prime,y_gold,disruptive,roc_area,loss


def mpi_train(conf,shot_list_train,shot_list_validate,loader, callbacks_list=None):   
    conf['num_workers'] = comm.Get_size()

    specific_builder = builder.ModelBuilder(conf)
    train_model = specific_builder.build_model(False)

    #load the latest epoch we did. Returns -1 if none exist yet
    e = specific_builder.load_model_weights(train_model)
    e_old = e

    num_epochs = conf['training']['num_epochs']
    lr_decay = conf['model']['lr_decay']
    batch_size = conf['training']['batch_size']
    lr = conf['model']['lr']
    warmup_steps = conf['model']['warmup_steps']
    num_batches_minimum = conf['training']['num_batches_minimum']

    if conf['model']['optimizer'] == 'adam':
        optimizer = MPIAdam(lr=lr)
    elif conf['model']['optimizer'] == 'sgd':
        optimizer = MPISGD(lr=lr)
    else:
        print("Optimizer not implemented yet")
        exit(1)

    print('{} epochs left to go'.format(num_epochs - 1 - e))

    # batch_generator = partial(loader.training_batch_generator,shot_list=shot_list_train)
    batch_generator = partial(loader.training_batch_generator_partial_reset,shot_list=shot_list_train)
    #{}batch_generator = partial(loader.training_batch_generator_process,shot_list=shot_list_train)

    print("warmup {}".format(warmup_steps))
    mpi_model = MPIModel(train_model,optimizer,comm,batch_generator,batch_size,lr=lr,warmup_steps = warmup_steps,num_batches_minimum=num_batches_minimum)
    mpi_model.compile(conf['model']['optimizer'],loss=conf['data']['target'].loss)

    tensorboard = None
    if backend != "theano" and task_index == 0:
        tensorboard_save_path = conf['paths']['tensorboard_save_path']
        write_grads = conf['callbacks']['write_grads']
        tensorboard = TensorBoard(log_dir=tensorboard_save_path,histogram_freq=1,write_graph=True,write_grads=write_grads)
        tensorboard.set_model(mpi_model.model)
        mpi_model.model.summary()

    if task_index == 0:
        callbacks = mpi_model.build_callbacks(conf,callbacks_list)
        callbacks.set_model(mpi_model.model)
        callback_metrics = conf['callbacks']['metrics']
        callbacks.set_params({
        'epochs': num_epochs,
        'metrics': callback_metrics,
        'batch_size': batch_size,
        })
        callbacks.on_train_begin()
    if conf['callbacks']['mode'] == 'max':
        best_so_far = -np.inf
        cmp_fn = max
    else:
        best_so_far = np.inf
        cmp_fn = min

    while e < num_epochs-1:
        if task_index == 0:
            callbacks.on_epoch_begin(int(round(e)))
        mpi_model.set_lr(lr*lr_decay**e)
        print_unique('\nEpoch {}/{}'.format(e,num_epochs))

        (step,ave_loss,curr_loss,num_so_far,effective_epochs) = mpi_model.train_epoch()
        e = e_old + effective_epochs

        loader.verbose=False #True during the first iteration
        if task_index == 0: 
            specific_builder.save_model_weights(train_model,int(round(e)))

        epoch_logs = {}
        
        _,_,_,roc_area,loss = mpi_make_predictions_and_evaluate(conf,shot_list_validate,loader)
        if conf['training']['ranking_difficulty_fac'] != 1.0:
            _,_,_,roc_area_train,loss_train = mpi_make_predictions_and_evaluate(conf,shot_list_train,loader)
            batch_generator = partial(loader.training_batch_generator_partial_reset,shot_list=shot_list_train)
            mpi_model.batch_iterator = batch_generator
            mpi_model.batch_iterator_func.__exit__()
            mpi_model.num_so_far_accum = mpi_model.num_so_far_indiv
            mpi_model.set_batch_iterator_func()
        epoch_logs['val_roc'] = roc_area 
        epoch_logs['val_loss'] = loss
        epoch_logs['train_loss'] = ave_loss
        best_so_far = cmp_fn(epoch_logs[conf['callbacks']['monitor']],best_so_far)

        stop_training = False
        if task_index == 0:
            print('=========Summary======== for epoch{}'.format(step))
            print('Training Loss numpy: {:.3e}'.format(ave_loss))
            print('Validation Loss: {:.3e}'.format(loss))
            print('Validation ROC: {:.4f}'.format(roc_area))
            if conf['training']['ranking_difficulty_fac'] != 1.0:
                print('Training Loss: {:.3e}'.format(loss_train))
                print('Training ROC: {:.4f}'.format(roc_area_train))

            callbacks.on_epoch_end(int(round(e)), epoch_logs)
            if hasattr(mpi_model.model,'stop_training'):
                stop_training = mpi_model.model.stop_training
            if best_so_far != epoch_logs[conf['callbacks']['monitor']]: #only save model weights if quantity we are tracking is improving
                specific_builder.delete_model_weights(train_model,int(round(e)))

            #tensorboard
            if backend != 'theano':
                val_generator = partial(loader.training_batch_generator,shot_list=shot_list_validate)()
                val_steps = 1
                tensorboard.on_epoch_end(val_generator,val_steps,int(round(e)),epoch_logs)

        stop_training = comm.bcast(stop_training,root=0)
        if stop_training:
            print("Stopping training due to early stopping")
            break

    if task_index == 0:
        callbacks.on_train_end()
        tensorboard.on_train_end()

    mpi_model.close()


def get_stop_training(callbacks):
    for cb in callbacks.callbacks:
        if isinstance(cb,cbks.EarlyStopping):
            print("Checking for early stopping")
            return cb.model.stop_training
    print("No early stopping callback found.")
    return False

class TensorBoard(object):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 validation_steps=0,
                 write_graph=True,
                 write_grads=False):
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.writer = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.validation_steps = validation_steps
        self.sess = None
        self.model = None

    def set_model(self, model):
        self.model = model
        print(type(self.model))
        self.sess = K.get_session()

        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = self.model.optimizer.get_gradients(self.model.total_loss,
                                                            weight)
                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        for grad in grads: 
                            tf.summary.histogram('{}_grad'.format(mapped_weight_name), grad)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                       layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                              self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)


    def on_epoch_end(self, val_generator, val_steps, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
            self.writer.flush()

        tensors = (self.model.inputs +
                   self.model.targets +
                   self.model.sample_weights)

        if self.model.uses_learning_phase:
            tensors += [K.learning_phase()]

        self.sess = K.get_session()

        for val_data in val_generator:
            batch_val = []
            sh = val_data[0].shape[0]
            batch_val.append(val_data[0])
            batch_val.append(val_data[1])
            batch_val.append(np.ones(sh))
            if self.model.uses_learning_phase:
                batch_val.append(1)

            feed_dict = dict(zip(tensors, batch_val))
            result = self.sess.run([self.merged], feed_dict=feed_dict)
            summary_str = result[0]
            self.writer.add_summary(summary_str, int(round(epoch)))
            val_steps -= 1
            if val_steps <= 0: break


    def on_train_end(self):
        self.writer.close()
