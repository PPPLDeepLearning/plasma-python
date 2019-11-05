from plasma.models.mpi_runner import (
    mpi_train, mpi_make_predictions_and_evaluate,
    comm, task_index, print_unique
    )
# from mpi4py import MPI
from plasma.preprocessor.preprocess import guarantee_preprocessed
from plasma.models.loader import Loader
from plasma.conf import conf
# from pprint import pprint
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

import os
import sys
import datetime
import random
import numpy as np

import matplotlib
matplotlib.use('Agg')

sys.setrecursionlimit(10000)


if conf['model']['shallow']:
    print("Shallow learning using MPI is not supported yet. ",
          "Set conf['model']['shallow'] to False.")
    exit(1)
if conf['data']['normalizer'] == 'minmax':
    from plasma.preprocessor.normalize import MinMaxNormalizer as Normalizer
elif conf['data']['normalizer'] == 'meanvar':
    from plasma.preprocessor.normalize import MeanVarNormalizer as Normalizer
elif conf['data']['normalizer'] == 'var':
    # performs !much better than minmaxnormalizer
    from plasma.preprocessor.normalize import VarNormalizer as Normalizer
elif conf['data']['normalizer'] == 'averagevar':
    # performs !much better than minmaxnormalizer
    from plasma.preprocessor.normalize import (
        AveragingVarNormalizer as Normalizer
        )
else:
    print('unkown normalizer. exiting')
    exit(1)

# TODO(KGF): this part of the code is duplicated in mpi_runner.py
# comm = MPI.COMM_WORLD
# task_index = comm.Get_rank()
# num_workers = comm.Get_size()

# NUM_GPUS = conf['num_gpus']
# MY_GPU = task_index % NUM_GPUS
# backend = conf['model']['backend']

# if task_index == 0:
#     pprint(conf)

# TODO(KGF): confirm that this second PRNG seed setting is not needed
# (before normalization; done again before MPI training)
# np.random.seed(task_index)
# random.seed(task_index)


only_predict = len(sys.argv) > 1
custom_path = None
if only_predict:
    custom_path = sys.argv[1]
    print_unique("predicting using path {}".format(custom_path))


#####################################################
#                 NORMALIZATION                     #
#####################################################
# make sure preprocessing has been run, and is saved as a file
if task_index == 0:
    # TODO(KGF): check tuple unpack
    (shot_list_train, shot_list_validate,
     shot_list_test) = guarantee_preprocessed(conf)
comm.Barrier()
(shot_list_train, shot_list_validate,
 shot_list_test) = guarantee_preprocessed(conf)


print_unique("normalization", end='')
normalizer = Normalizer(conf)
normalizer.train()
loader = Loader(conf, normalizer)
print_unique("...done")

# ensure training has a separate random seed for every worker
np.random.seed(task_index)
random.seed(task_index)
if not only_predict:
    mpi_train(conf, shot_list_train, shot_list_validate, loader,
              shot_list_test=shot_list_test)

# load last model for testing
loader.set_inference_mode(True)
print_unique('saving results')
y_prime = []
y_gold = []
disruptive = []

# TODO(KGF): check tuple unpack
(y_prime_train, y_gold_train, disruptive_train, roc_train,
 loss_train) = mpi_make_predictions_and_evaluate(conf, shot_list_train,
                                                 loader, custom_path)
(y_prime_test, y_gold_test, disruptive_test, roc_test,
 loss_test) = mpi_make_predictions_and_evaluate(conf, shot_list_test,
                                                loader, custom_path)

print_unique('=========Summary========')
print_unique('Train Loss: {:.3e}'.format(loss_train))
print_unique('Train ROC: {:.4f}'.format(roc_train))
print_unique('Test Loss: {:.3e}'.format(loss_test))
print_unique('Test ROC: {:.4f}'.format(roc_test))

if task_index == 0:
    disruptive_train = np.array(disruptive_train)
    disruptive_test = np.array(disruptive_test)

    y_gold = y_gold_train + y_gold_test
    y_prime = y_prime_train + y_prime_test
    disruptive = np.concatenate((disruptive_train, disruptive_test))

    shot_list_test.make_light()
    shot_list_train.make_light()

    save_str = 'results_' + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H-%M-%S")
    result_base_path = conf['paths']['results_prepath']
    if not os.path.exists(result_base_path):
        os.makedirs(result_base_path)

    np.savez(result_base_path+save_str, y_gold=y_gold,
             y_gold_train=y_gold_train, y_gold_test=y_gold_test,
             y_prime=y_prime, y_prime_train=y_prime_train,
             y_prime_test=y_prime_test, disruptive=disruptive,
             disruptive_train=disruptive_train,
             disruptive_test=disruptive_test, shot_list_train=shot_list_train,
             shot_list_test=shot_list_test, conf=conf)

    # TODO(KGF): Intel NumPy fork
    # https://conda.anaconda.org/intel/linux-64/numpy-1.16.2-py36h7b7c402_0.tar.bz2
    # applies cve_2019_6446_fix.patch, which unlike main NumPy, adds
    # requirement for "allow_pickle=True" to savez() calls

sys.stdout.flush()
print_unique('finished.')
