#!/bin/bash
#SBATCH -t 01:30:00
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH -c 4


module load anaconda
source activate PPPL
module load cudatoolkit/8.0 cudann/cuda-8.0/5.1 openmpi/intel-17.0/1.10.2/64 intel/17.0/64/17.0.2.174

rm /tigress/alexeys/model_checkpoints/*
rm /tigress/alexeys/csv_logs/*
rm /tigress/alexeys/Graph/*

srun python mpi_learn.py
