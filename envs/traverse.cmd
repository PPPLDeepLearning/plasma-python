#!/usr/bin/env bash

module load anaconda3
conda activate frnn

module load cudatoolkit/11.0
module load cudnn/cuda-10.1/7.6.1

# after RHEL 8 upgrade
module load openmpi/gcc/4.0.4/64
module load hdf5/gcc/openmpi-4.0.4/1.10.6
