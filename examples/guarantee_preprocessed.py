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
from plasma.preprocessor.preprocess import guarantee_preprocessed

os.environ["PYTHONHASHSEED"] = 0

#####################################################
####################PREPROCESSING####################
#####################################################
np.random.seed(0)
random.seed(0)
guarantee_preprocessed(conf)
