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
import random
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functools import partial
import itertools
import socket
sys.setrecursionlimit(10000)
import getpass

from plasma.conf import conf
from pprint import pprint
pprint(conf)
from plasma.preprocessor.normalize import Normalizer
from plasma.preprocessor.preprocess import Preprocessor
from plasma.models.loader import Loader

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
NUM_GPUS = 4
MY_GPU = task_index % NUM_GPUS

#####################################################
####################PREPROCESSING####################
#####################################################
if task_index ==0:
    print("preprocessing all shots",end='')
    pp = Preprocessor(conf)
    pp.clean_shot_lists()
    shot_list = pp.preprocess_all()
    sorted(shot_list)
    shot_list_train,shot_list_test = shot_list.split_train_test(conf)
    num_shots = len(shot_list_train) + len(shot_list_test)
    validation_frac = conf['training']['validation_frac']
    if validation_frac <= 0.0:
        print('Setting validation to a minimum of 0.05')
        validation_frac = 0.05
    shot_list_train,shot_list_validate = shot_list_train.split_direct(1.0-validation_frac,do_shuffle=True)
    print('validate: {} shots, {} disruptive'.format(len(shot_list_validate),shot_list_validate.num_disruptive()))
    print('training: {} shots, {} disruptive'.format(len(shot_list_train),shot_list_train.num_disruptive()))
    print('testing: {} shots, {} disruptive'.format(len(shot_list_test),shot_list_test.num_disruptive()))
    print("...done")

    pp.save_shotlists(conf,shot_list_train,shot_list_validate,shot_list_test)

    #####################################################
    ####################Normalization####################
    #####################################################

    print("normalization",end='')
    nn = Normalizer(conf)
    nn.train()
    loader = Loader(conf,nn)
    print("...done")

from plasma.models.mpi_runner import *

np.random.seed(task_index)
random.seed(task_index)

shot_list_train,shot_list_validate,shot_list_test = loader.load_shotlists(conf)

mpi_train(conf,shot_list_train,shot_list_validate,loader)

#load last model for testing
print('saving results')
y_prime = []
y_prime_test = []
y_prime_train = []

y_gold = []
y_gold_test = []
y_gold_train = []

disruptive= []
disruptive_train= []
disruptive_test= []

# y_prime_train,y_gold_train,disruptive_train = make_predictions(conf,shot_list_train,loader)
# y_prime_test,y_gold_test,disruptive_test = make_predictions(conf,shot_list_test,loader)

y_prime_train,y_gold_train,disruptive_train = mpi_make_predictions(conf,shot_list_train,loader)
y_prime_test,y_gold_test,disruptive_test = mpi_make_predictions(conf,shot_list_test,loader)


if task_index == 0:
    disruptive_train = np.array(disruptive_train)
    disruptive_test = np.array(disruptive_test)

    y_gold = y_gold_train + y_gold_test
    y_prime = y_prime_train + y_prime_test
    disruptive = np.concatenate((disruptive_train,disruptive_test))

    shot_list_test.make_light()
    shot_list_train.make_light()

    save_str = 'results_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    np.savez(conf['paths']['results_prepath']+save_str,
        y_gold=y_gold,y_gold_train=y_gold_train,y_gold_test=y_gold_test,
        y_prime=y_prime,y_prime_train=y_prime_train,y_prime_test=y_prime_test,
        disruptive=disruptive,disruptive_train=disruptive_train,disruptive_test=disruptive_test,
        shot_list_train=shot_list_train,shot_list_test=shot_list_test,
        conf = conf)

    print('finished.')
