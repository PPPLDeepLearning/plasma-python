#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -o 0.out


module load anaconda
module load cudatoolkit/8.0 cudann/cuda-8.0/5.1 openmpi/intel-17.0/1.10.2/64 intel/17.0/64/17.0.2.174
rm -f /tigress/jk7/model_checkpoints/*
srun python mpi_learn.py
echo "done."
