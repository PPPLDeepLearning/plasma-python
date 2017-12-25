#!/bin/bash

rm /tigress/alexeys/model_checkpoints/*
rm -rf /tigress/alexeys/processed_shots
rm -rf /tigress/alexeys/processed_shotlists
rm -rf /tigress/alexeys/normalization


ls ${PWD}

echo "Jenkins test Python3.6"
export PYTHONHASHSEED=0
module load anaconda3
source activate PPPL_dev3
module load cudatoolkit/8.0
module load cudnn/cuda-8.0/6.0
module load openmpi/cuda-8.0/intel-17.0/2.1.0/64
module load intel/17.0/64/17.0.4.196
export OMPI_MCA_btl="tcp,self,sm"

echo $SLURM_NODELIST
srun python mpi_learn.py
