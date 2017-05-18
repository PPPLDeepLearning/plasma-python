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
from mpi4py import MPI
comm = MPI.COMM_WORLD
task_index = comm.Get_rank()
num_workers = comm.Get_size()
NUM_GPUS = 4
MY_GPU = task_index % NUM_GPUS

from pprint import pprint
from plasma.conf import conf

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
    from keras.optimizers import SGD
    from keras.utils.generic_utils import Progbar 
    import keras.callbacks as cbks

from plasma.models import builder
from plasma.utils.evaluation import get_loss_from_list
from plasma.utils.processing import concatenate_sublists
from plasma.utils.performance import PerformanceAnalyzer

if task_index == 0:
    pprint(conf)



###TODO add optimizers other than SGD


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

class MPIAdam(MPIOptimizer):
  def __init__(self,lr):
    super(MPIAdam,self).__init__(lr)
    self.beta_1 = 0.9
    self.beta_2 = 0.999
    self.eps = 1e-8

  def get_deltas(self,raw_deltas):

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
  def __init__(self,model,optimizer,comm,batch_iterator,batch_size,num_replicas=None,warmup_steps=1000,lr=0.01):
    # random.seed(task_index)
    self.epoch = 0
    self.num_so_far = 0
    self.num_so_far_accum = 0
    self.num_so_far_indiv = 0
    self.model = model
    self.optimizer = optimizer
    self.max_lr = 0.1
    self.lr = lr if (lr < self.max_lr) else self.max_lr
    self.DUMMY_LR = 0.1
    self.comm = comm
    self.batch_size = batch_size
    self.batch_iterator_func = batch_iterator()
    self.batch_iterator = batch_iterator
    self.warmup_steps=warmup_steps
    self.num_workers = comm.Get_size()
    self.task_index = comm.Get_rank()
    self.history = cbks.History()
    if num_replicas is None or num_replicas < 1 or num_replicas > self.num_workers:
        self.num_replicas = self.num_workers
    else:
        self.num_replicas = num_replicas

  def set_lr(self,lr):
    self.lr = lr

  def save_weights(self,path,overwrite=False):
    self.model.save_weights(path,overwrite=overwrite)

  def load_weights(self,path):
    self.model.load_weights(path)

  def compile(self,loss='mse'):
    self.model.compile(optimizer=SGD(lr=self.DUMMY_LR),loss=loss)


  def get_deltas(self,X_batch,Y_batch,verbose=False):
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

    if comm.rank == 0:
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
      callback_save_path = conf['paths']['callback_save_path']
      callbacks_list = conf['callbacks']['list']

      callbacks = [cbks.BaseLogger()]
      callbacks += [self.history]
      callbacks += [cbks.CSVLogger("{}callbacks-{}.log".format(callback_save_path,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))]

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

    num_batches_minimum = 100
    num_batches_current = 0

    while (self.num_so_far-self.epoch*num_total) < num_total or num_batches_current < num_batches_minimum:

      try:
          batch_xs,batch_ys,reset_states_now,num_so_far_curr,num_total = next(batch_iterator_func)
      except StopIteration:
	  print("Resetting batch iterator.")
          self.num_so_far_accum = self.num_so_far_indiv
          batch_iterator_func = self.batch_iterator()
          batch_xs,batch_ys,reset_states_now,num_so_far_curr,num_total = next(batch_iterator_func)
      self.num_so_far_indiv = self.num_so_far_accum+num_so_far_curr

      num_batches_current +=1 

      if reset_states_now:
        self.model.reset_states()

      warmup_phase = (step < self.warmup_steps and self.epoch == 0)
      num_replicas = 1 if warmup_phase else self.num_replicas

      self.num_so_far = self.mpi_sum_scalars(self.num_so_far_indiv,num_replicas)

      #run the model once to force compilation. Don't actually use these values.
      if step == 0 and self.epoch == 0:
        t0_comp = time.time()
        _,_ = self.get_deltas(batch_xs,batch_ys,verbose)
        comm.Barrier()
        sys.stdout.flush()
        print_unique('Compilation finished in {:.2f}s'.format(time.time()-t0_comp))
        t_start = time.time()
        sys.stdout.flush()  

      t0 = time.time()
      deltas,loss = self.get_deltas(batch_xs,batch_ys,verbose)
      t1 = time.time()
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

def mpi_make_predictions(conf,shot_list,loader):
    shot_list.sort()#make sure all replicas have the same list
    specific_builder = builder.ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []

    model = specific_builder.build_model(True)
    specific_builder.load_model_weights(model)
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
            y_p = model.predict(X,
                batch_size=conf['model']['pred_batch_size'])
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


def mpi_make_predictions_and_evaluate(conf,shot_list,loader):
    y_prime,y_gold,disruptive = mpi_make_predictions(conf,shot_list,loader)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'].loss)
    return y_prime,y_gold,disruptive,roc_area,loss


def mpi_train(conf,shot_list_train,shot_list_validate,loader, callbacks_list=None):   

    specific_builder = builder.ModelBuilder(conf)
    train_model = specific_builder.build_model(False)

    #load the latest epoch we did. Returns -1 if none exist yet
    e = specific_builder.load_model_weights(train_model)

    num_epochs = conf['training']['num_epochs']
    lr_decay = conf['model']['lr_decay']
    batch_size = conf['training']['batch_size']
    lr = conf['model']['lr']
    warmup_steps = conf['model']['warmup_steps']
    optimizer = MPIAdam(lr=lr)
    print('{} epochs left to go'.format(num_epochs - 1 - e))

    batch_generator = partial(loader.training_batch_generator,shot_list=shot_list_train)

    mpi_model = MPIModel(train_model,optimizer,comm,batch_generator,batch_size,lr=lr,warmup_steps = warmup_steps)
    mpi_model.compile(loss=conf['data']['target'].loss)

    callbacks = mpi_model.build_callbacks(conf,callbacks_list)

    callbacks.set_model(mpi_model.model)
    callback_metrics = conf['callbacks']['metrics']

    callbacks.set_params({
        'epochs': num_epochs,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    while e < num_epochs-1:
        callbacks.on_epoch_begin(int(round(e)))
        mpi_model.set_lr(lr*lr_decay**e)
        print_unique('\nEpoch {}/{}'.format(e,num_epochs))

        (step,ave_loss,curr_loss,num_so_far,effective_epochs) = mpi_model.train_epoch()
        e = effective_epochs

        loader.verbose=False #True during the first iteration
        if task_index == 0:
            specific_builder.save_model_weights(train_model,int(round(e)))

        epoch_logs = {}
        _,_,_,roc_area,loss = mpi_make_predictions_and_evaluate(conf,shot_list_validate,loader)

        #validation_losses.append(loss)
        #validation_roc.append(roc_area)
        #training_losses.append(ave_loss)

        epoch_logs['val_roc'] = roc_area 
        epoch_logs['val_loss'] = loss
        epoch_logs['train_loss'] = ave_loss

        if task_index == 0:
            print('=========Summary======== for epoch{}'.format(step))
            print('Training Loss: {:.3e}'.format(ave_loss))
            print('Validation Loss: {:.3e}'.format(loss))
            print('Validation ROC: {:.4f}'.format(roc_area))

            callbacks.on_epoch_end(int(round(e)), epoch_logs)

    callbacks.on_train_end()
