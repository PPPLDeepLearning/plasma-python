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
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functools import partial
import itertools
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
backend = 'theano'

from plasma.models import builder
from plasma.conf import conf
from plasma.utils.evaluation import get_loss_from_list
from pprint import pprint
if task_index == 0:
    pprint(conf)
from plasma.utils.processing import concatenate_sublists
from plasma.utils.performance import PerformanceAnalyzer

if backend == 'tf' or backend == 'tensorflow':
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(MY_GPU)#,mode=NanGuardMode'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    import tensorflow
else:
    base_compile_dir = '/scratch/{}/tmp/{}-{}'.format(getpass.getuser(),socket.gethostname(),task_index)
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
    self.model = model
    self.optimizer = optimizer
    self.lr = lr
    self.DUMMY_LR = 0.1
    self.max_lr = 0.1
    self.comm = comm
    self.batch_size = batch_size
    self.batch_iterator = batch_iterator
    self.warmup_steps=warmup_steps
    self.num_workers = comm.Get_size()
    self.task_index = comm.Get_rank()
    if num_replicas is None or num_replicas < 1 or num_replicas > self.num_workers:
      self.num_replicas = num_workers
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
    val_global = self.mpi_sum_scalars(val,num_replicas)
    val_global /= num_replicas
    return val_global


  def mpi_sum_scalars(self,val,num_replicas=None):
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



  def train_epoch(self):
    verbose = False
    step = 0
    loss_averager = Averager()
    t_start = time.time()
    for batch_xs,batch_ys,reset_states_now,num_so_far,num_total in self.batch_iterator():

      if reset_states_now:
        self.model.reset_states()

      warmup_phase = (step < self.warmup_steps and self.epoch == 0)
      num_replicas = 1 if warmup_phase else self.num_replicas

      num_so_far = self.mpi_sum_scalars(num_so_far,num_replicas)
      epoch_end = num_so_far >= num_total

      t0 = time.time()
      deltas,loss = self.get_deltas(batch_xs,batch_ys,verbose)
      t1 = time.time()
      self.set_new_weights(deltas,num_replicas)
      t2 = time.time()
      write_str_0 = self.calculate_speed(t0,t1,t2,num_replicas)

      curr_loss = self.mpi_average_scalars(1.0*loss,num_replicas)
      loss_averager.add_val(curr_loss)
      ave_loss = loss_averager.get_val()
      eta = self.estimate_remaining_time(t0 - t_start,num_so_far,num_total)
      write_str = '\r[{}] step: {} [ETA: {:.2f}s] [{:.2f}/{}], loss: {:.5f} [{:.5f}] | '.format(self.task_index,step,eta,1.0*num_so_far,num_total,ave_loss,curr_loss)
      print_unique(write_str + write_str_0)
      step += 1
      if epoch_end:
        self.epoch += 1
        print_unique('\nEpoch {} finished in {:.2f} seconds.\n'.format(self.epoch,t2 - t_start))
        break


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

  def train_epochs(self,warmup_steps=100,num_epochs=1):
    for i in range(num_epochs):
      self.train_epoch(warmup_steps)


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

    specific_builder = builder.ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []

    _,model = specific_builder.build_train_test_models()
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
    return roc_area,loss




def mpi_train(conf,shot_list_train,shot_list_validate,loader):
    validation_losses = []
    validation_roc = []
    # training_losses = []

    specific_builder = builder.ModelBuilder(conf)
    train_model,test_model = specific_builder.build_train_test_models()

    #load the latest epoch we did. Returns -1 if none exist yet
    e = specific_builder.load_model_weights(train_model)

    num_epochs = conf['training']['num_epochs']
    lr_decay = conf['model']['lr_decay']
    batch_size = conf['training']['batch_size']
    lr = conf['model']['lr']
    optimizer = MPIAdam(lr=lr)
    print('{} epochs left to go'.format(num_epochs - 1 - e))
    batch_generator = partial(loader.training_batch_generator,shot_list=shot_list_train,loader=loader)

    mpi_model = MPIModel(train_model,optimizer,comm,batch_generator,batch_size,lr=lr,warmup_steps = 50)
    mpi_model.compile(loss=conf['data']['target'].loss)


    while e < num_epochs-1:
        e += 1
        mpi_model.set_lr(lr*lr_decay**e)
        print_unique('\nEpoch {}/{}'.format(e,num_epochs))

        mpi_model.train_epoch()

        loader.verbose=False#True during the first iteration
        if task_index == 0:
            specific_builder.save_model_weights(train_model,e)


        roc_area,loss = mpi_make_predictions_and_evaluate(conf,shot_list_validate,loader)

        validation_losses.append(loss)
        validation_roc.append(roc_area)

        if task_index == 0:
            print('=========Summary========')
            # print('Training Loss: {:.3e}'.format(training_losses[-1]))
            print('Validation Loss: {:.3e}'.format(validation_losses[-1]))
            print('Validation ROC: {:.4f}'.format(validation_roc[-1]))

#mpi_train(conf,shot_list_train,shot_list_validate,loader)

#load last model for testing
