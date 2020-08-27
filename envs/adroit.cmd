#!/usr/bin/env bash

module load anaconda3
# must activate conda env before module loads
conda activate frnn
export OMPI_MCA_btl="tcp,self,vader"

module load cudatoolkit/10.2
module load cudnn/cuda-10.1/7.6.3
module load openmpi/gcc/3.1.5/64
module load hdf5/gcc/openmpi-1.10.2/1.10.0 # like TigerGPU, this is older than version on Traverse, hdf5/gcc/openmpi-3.1.4/1.10.5
