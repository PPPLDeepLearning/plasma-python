'''
#########################################################
This file trains a deep learning model to predict
disruptions on time series data from plasma discharges.

Must run guarantee_preprocessed.py in order for this to work.

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
import random
import numpy as np
import copy
from functools import partial

import matplotlib
matplotlib.use('Agg')

from pprint import pprint
sys.setrecursionlimit(10000)

from plasma.conf import conf
from plasma.models.loader import Loader
from plasma.primitives.shots import ShotList
from plasma.preprocessor.normalize import Normalizer
from plasma.preprocessor.augment import ByShotAugmentator
from plasma.preprocessor.preprocess import guarantee_preprocessed

if conf['model']['shallow']:
    print("Shallow learning using MPI is not supported yet. set conf['model']['shallow'] to false.")
    exit(1)
if conf['data']['normalizer'] == 'minmax':
    from plasma.preprocessor.normalize import MinMaxNormalizer as Normalizer
elif conf['data']['normalizer'] == 'meanvar':
    from plasma.preprocessor.normalize import MeanVarNormalizer as Normalizer
elif conf['data']['normalizer'] == 'var':
    from plasma.preprocessor.normalize import VarNormalizer as Normalizer #performs !much better than minmaxnormalizer
elif conf['data']['normalizer'] == 'averagevar':
    from plasma.preprocessor.normalize import AveragingVarNormalizer as Normalizer #performs !much better than minmaxnormalizer
else:
    print('unkown normalizer. exiting')
    exit(1)

from mpi4py import MPI
comm = MPI.COMM_WORLD
task_index = comm.Get_rank()
num_workers = comm.Get_size()
NUM_GPUS = conf['num_gpus']
MY_GPU = task_index % NUM_GPUS

from plasma.models.mpi_runner import *

np.random.seed(task_index)
random.seed(task_index)
if task_index == 0:
    pprint(conf)

only_predict = len(sys.argv) > 1
custom_path = None
if only_predict:
    custom_path = sys.argv[1]
print("predicting using path {}".format(custom_path))

assert(only_predict)
#####################################################
####################Normalization####################
#####################################################
if task_index == 0: #make sure preprocessing has been run, and is saved as a file
    shot_list_train,shot_list_validate,shot_list_test = guarantee_preprocessed(conf)
comm.Barrier()
shot_list_train,shot_list_validate,shot_list_test = guarantee_preprocessed(conf)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    return[ l[i:i + n] for i in range(0, len(l), n)]

def hide_signal_data(shot,t=0,sigs_to_hide=None):
    for sig in shot.signals:
        if sigs_to_hide is None or (sigs_to_hide is not None and sig in sigs_to_hide):
            shot.signals_dict[sig][t:,:] = shot.signals_dict[sig][t,:]

def create_shot_list_tmp(original_shot,time_points,sigs=None):
    shot_list_tmp = ShotList()
    T = len(original_shot.ttd)
    t_range = np.linspace(0,T-1,time_points,dtype=np.int)
    for t in t_range:
        new_shot = copy.copy(original_shot)
        assert(new_shot.augmentation_fn == None)
        new_shot.augmentation_fn = partial(hide_signal_data,t = t,sigs_to_hide=sigs)
        #new_shot.number = original_shot.number
        shot_list_tmp.append(new_shot)
    return shot_list_tmp,t_range

def get_importance_measure(original_shot,loader,custom_path,metric,time_points=10,sig=None):
    shot_list_tmp,t_range = create_shot_list_tmp(original_shot,time_points,sigs) 
    y_prime,y_gold,disruptive = mpi_make_predictions(conf,shot_list_tmp,loader,custom_path)
    shot_list_tmp.make_light()
    return t_range,get_importance_measure_given_y_prime(y_prime,metric),y_prime[-1]

def difference_metric(y_prime,y_prime_orig):
    idx = np.argmax(y_prime_orig) 
    return (np.max(y_prime_orig) - y_prime[idx])/(np.max(y_prime_orig) - np.min(y_prime_orig))

def get_importance_measure_given_y_prime(y_prime,metric):
    differences = [metric(y_prime[i],y_prime[-1]) for i in range(len(y_prime))]
    return 1.0-np.array(differences)#/np.max(differences)


print("normalization",end='')
normalizer = Normalizer(conf)
normalizer.train()
normalizer = ByShotAugmentator(normalizer)
loader = Loader(conf,normalizer)
print("...done")

# if not only_predict:
#     mpi_train(conf,shot_list_train,shot_list_validate,loader)

#load last model for testing
loader.set_inference_mode(True)
use_signals = copy.copy(conf['paths']['use_signals'])
use_signals.append(None)



for shot in shot_list_test:
    shot.augmentation_fn = None# partial(hide_signal_data,t = 0,sigs_to_hide = sigs_to_hide)

print("All signals:")
y_prime,y_gold,disruptive,roc,loss = mpi_make_predictions_and_evaluate(conf,shot_list_test,loader,custom_path)
print(roc)
print(loss)

#for sigs_to_hide in [[s] for s in use_signals[:-3]] + [use_signals[-3:-1]] + [use_signals[-1]]:
for sigs_to_hide in [[s] for s in use_signals[:-3]] + [[s] for s in use_signals[-3:-1]] + [use_signals[-3:-1]]:# + [use_signals[-1]]:
    for shot in shot_list_test:
        shot.augmentation_fn = partial(hide_signal_data,t = 0,sigs_to_hide = sigs_to_hide)
    print("Hiding: {}".format(sigs_to_hide))
    y_prime,y_gold,disruptive,roc,loss = mpi_make_predictions_and_evaluate(conf,shot_list_test,loader,custom_path)
    print(roc)
    print(loss)



if task_index == 0:
    print('finished.')
