#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4


module load anaconda
source activate PPPL
module load cudatoolkit/7.5 cudann openmpi/intel-16.0/1.8.8/64
srun python mpi_learn.py
