import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

#define a float16 mpi datatype
mpi_float16 = MPI.BYTE.Create_contiguous(2).Commit()
MPI._typedict['e'] = mpi_float16

def sum_f16_cb(buffer_a, buffer_b, t):
    assert t == mpi_float16
    array_a = np.frombuffer(buffer_a, dtype='float16')
    array_b = np.frombuffer(buffer_b, dtype='float16')
    array_b += array_a

#create new OP
mpi_sum_f16 = MPI.Op.Create(sum_f16_cb, commute=True)
