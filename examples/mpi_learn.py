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

import matplotlib
matplotlib.use('Agg')

from pprint import pprint
sys.setrecursionlimit(10000)

from plasma.conf import conf
from plasma.models.loader import Loader
from plasma.preprocessor.normalize import Normalizer

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

from plasma.models.mpi_runner import *

np.random.seed(task_index)
random.seed(task_index)
if task_index == 0:
    pprint(conf)

#####################################################
####################Normalization####################
#####################################################
print("normalization",end='')
nn = Normalizer(conf)
nn.train()
loader = Loader(conf,nn)
print("...done")


shot_list_train,shot_list_validate,shot_list_test = loader.load_shotlists(conf)

mpi_train(conf,shot_list_train,shot_list_validate,loader)

#load last model for testing
print('saving results')
y_prime = []
y_gold = []
disruptive= []

# y_prime_train,y_gold_train,disruptive_train = make_predictions(conf,shot_list_train,loader)
# y_prime_test,y_gold_test,disruptive_test = make_predictions(conf,shot_list_test,loader)

y_prime_train,y_gold_train,disruptive_train,roc_train,loss_train = mpi_make_predictions_and_evaluate(conf,shot_list_train,loader)
y_prime_test,y_gold_test,disruptive_test,roc_test,loss_test = mpi_make_predictions_and_evaluate(conf,shot_list_test,loader)

if task_index == 0:
	print('=========Summary========')
	print('Train Loss: {:.3e}'.format(loss_train))
	print('Train ROC: {:.4f}'.format(roc_train))
	print('Test Loss: {:.3e}'.format(loss_test))
	print('Test ROC: {:.4f}'.format(roc_test))




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

    sys.stdout.flush()
    print('finished.')
