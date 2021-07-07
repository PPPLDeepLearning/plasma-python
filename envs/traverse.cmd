#!/usr/bin/env bash

module load anaconda3
conda activate frnn

module load cudatoolkit/11.3
module load cudnn/cuda-11.x/8.2.0

# after RHEL 8 upgrade
module load openmpi/gcc/4.0.4/64
module load hdf5/gcc/openmpi-4.0.4/1.10.6
