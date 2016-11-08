#!/bin/bash
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks-per-socket=8
#SBATCH --gres=gpu:4

module load anaconda
source activate PPPL
module load cudatoolkit/7.5 cudann
python mpi_learn.py
