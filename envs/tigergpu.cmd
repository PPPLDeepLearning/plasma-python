#!/usr/bin/env bash

module load anaconda3
# must activate conda env before module loads
conda activate frnn
export OMPI_MCA_btl="tcp,self,vader"  #sm"
module load cudatoolkit
module load cudnn

module load openmpi/gcc/3.1.3/64
module load hdf5/gcc/openmpi-1.10.2/1.10.0
