#!/usr/bin/env bash

module load anaconda3
# must activate conda env before module loads
conda activate frnn
export OMPI_MCA_btl="tcp,self,vader"

module load cudatoolkit
module load cudnn/cuda-10.1/7.6.1
module load openmpi/gcc/3.1.4/64
module load hdf5/gcc/openmpi-3.1.4/1.10.5
