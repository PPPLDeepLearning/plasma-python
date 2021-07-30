#!/bin/bash
#SBATCH --job-name=FRNNTest
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH --mem-per-cpu=0
#SBATCH --reservation test
#SBATCH --mail-user=jrodrig@stanford.edu
#SBATCH --mail-type=ALL

# Load modules
module load anaconda3/2020.7
conda activate FRNN
module load cudatoolkit/11.3
module load cudnn/cuda-11.x/8.2.0
module load openmpi/cuda-11.0/gcc/4.0.4/64
module load hdf5/gcc/openmpi-4.0.4/1.10.6

# remove checkpoints for a benchmark run
rm /tigress/$USER/model_checkpoints/*
rm /tigress/$USER/results/*
rm /tigress/$USER/csv_logs/*
rm /tigress/$USER/Graph/*
rm /tigress/$USER/normalization/*

srun python mpi_learn.py
