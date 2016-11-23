from __future__ import print_function
import os
import sys 
import time
import datetime
import random
import numpy as np

from plasma.conf import conf
from pprint import pprint
pprint(conf)
from plasma.preprocessor.preprocess import Preprocessor

#####################################################
####################PREPROCESSING####################
#####################################################
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
