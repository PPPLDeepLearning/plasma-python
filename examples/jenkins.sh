#!/bin/bash

rm /tigress/alexeys/model_checkpoints/*

ls ${PWD}

module load anaconda
module load cudatoolkit/8.0
module load openmpi/intel-17.0/2.1.0/64 intel/17.0/64/17.0.4.196 intel-mkl/2017.3/4/64
module load cudnn/cuda-8.0/6.0
source activate PPPL

export OMPI_MCA_btl="tcp,self,sm"

echo $SLURM_NODELIST
srun python mpi_learn.py
