import plasma.global_vars as g
g.init_MPI()
from plasma.models.mpi_runner import (
    mpi_train, mpi_make_predictions_and_evaluate
    )
from plasma.preprocessor.preprocess import guarantee_preprocessed
from plasma.models.loader import Loader
from plasma.conf import conf
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

# set PRNG seed, unique for each worker, based on MPI task index for
# reproducible shuffling in guranteed_preprocessed() and training steps
np.random.seed(g.task_index)
random.seed(g.task_index)

only_predict = len(sys.argv) > 1
custom_path = None
if only_predict:
    custom_path = sys.argv[1]
    g.print_unique("predicting using path {}".format(custom_path))

#####################################################
#                 NORMALIZATION                     #
#####################################################
normalizer = Normalizer(conf)
if g.task_index == 0:
    # make sure preprocessing has been run, and results are saved to files
    # if not, only master MPI rank spawns thread pool to perform preprocessing
    (shot_list_train, shot_list_validate,
     shot_list_test) = guarantee_preprocessed(conf)
    # similarly, train normalizer (if necessary) w/ master MPI rank only
    normalizer.train()  # verbose=False only suppresses if purely loading
g.comm.Barrier()
g.print_unique("begin preprocessor+normalization (all MPI ranks)...")
# second call has ALL MPI ranks load preprocessed shots from .npz files
(shot_list_train, shot_list_validate,
 shot_list_test) = guarantee_preprocessed(conf, verbose=True)
# second call to normalizer training
normalizer.conf['data']['recompute_normalization'] = False
normalizer.train(verbose=True)
# KGF: may want to set it back...
# normalizer.conf['data']['recompute_normalization'] = conf['data']['recompute_normalization']   # noqa
loader = Loader(conf, normalizer)
g.print_unique("...done")

# TODO(KGF): both preprocess.py and normalize.py are littered with print()
# calls that should probably be replaced with print_unique() when they are not
# purely loading previously-computed quantities from file
# (or we can continue to ensure that they are only ever executed by 1 rank)

#####################################################
#                    TRAINING                       #
#####################################################

# Prevent Keras TF backend deprecation messages from mpi_train() from
# appearing jumbled with stdout, stderr msgs from above steps
g.comm.Barrier()
g.flush_all_inorder()

# reminder: ensure training has a separate random seed for every worker
if not only_predict:
    mpi_train(conf, shot_list_train, shot_list_validate, loader,
              shot_list_test=shot_list_test)
g.flush_all_inorder()

#####################################################
#                    TESTING                        #
#####################################################

# load last model for testing
loader.set_inference_mode(True)
g.print_unique('saving results')
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

g.print_unique('=========Summary========')
g.print_unique('Train Loss: {:.3e}'.format(loss_train))
g.print_unique('Train ROC: {:.4f}'.format(roc_train))
g.print_unique('Test Loss: {:.3e}'.format(loss_test))
g.print_unique('Test ROC: {:.4f}'.format(roc_test))

if g.task_index == 0:
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
g.print_unique('finished.')
