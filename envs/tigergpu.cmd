#!/usr/bin/env bash

module load anaconda3  # KGF: issue with my shell--- makes conda CLI return nothing
# must activate conda env before module loads to ensure MPI, etc modules have
# precedence for setting the environment variables, libraries, etc.
conda activate frnn
export OMPI_MCA_btl="tcp,self,vader"  #sm"
module load cudatoolkit
module load cudnn

module load openmpi/gcc/3.1.3/64
module load hdf5/gcc/openmpi-1.10.2/1.10.0 # like Adroit, this is older than version on Traverse, hdf5/gcc/openmpi-3.1.4/1.10.5
