from plasma.preprocessor.preprocess import guarantee_preprocessed
import random
import numpy as np

from plasma.conf import conf
from pprint import pprint
pprint(conf)

#####################################################
#                PREPROCESSING                      #
#####################################################
np.random.seed(0)
random.seed(0)
guarantee_preprocessed(conf, verbose=True)
