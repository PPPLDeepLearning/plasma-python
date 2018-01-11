#!/usr/bin/env python

import mpi4py as mmm
print(mmm.__version__)

import keras as kk
print(kk.__version__)

import tensorflow as tf
print(tf.__version__)

from mpi4py import MPI
import sys

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

sys.stdout.write(
    "Hello, World! I am process %d of %d on %s.\n"
    % (rank, size, name))
