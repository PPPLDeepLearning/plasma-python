#!/usr/bin/env python

import sys
from mpi4py import MPI
import tensorflow as tf
import keras as kk
import mpi4py as mmm

print(mmm.__version__)
print(kk.__version__)
print(tf.__version__)

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

sys.stdout.write(
    "Hello, World! I am process %d of %d on %s.\n"
    % (rank, size, name))
