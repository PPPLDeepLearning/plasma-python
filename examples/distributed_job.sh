#!/bin/bash
#SBATCH -t 0-2:00
#SBATCH -N 5
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks-per-socket=8
#SBATCH --gres=gpu:4

module load anaconda
module load cudatoolkit/7.5 cudann
module load openmpi

echo "Removing old model checkpoints."
rm /tigress/jk7/data/model_checkpoints/*
echo "Running distributed learning"
mpirun -npernode 4 python mpi_learn.py

