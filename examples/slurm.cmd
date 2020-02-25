#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH --mem-per-cpu=0

# Example Slurm configuration for TigerGPU nodes (4 nodes, 16 GPUs total)
# Each node = 2.4 GHz Xeon Broadwell E5-2680 v4 + 4x 1328 MHz P100 GPU

module load anaconda3
conda activate my_env
module load cudatoolkit
module load cudnn
module load openmpi/cuda-8.0/intel-17.0/3.0.0/64
module load intel/19.0/64/19.0.3.199
module load hdf5/intel-17.0/intel-mpi/1.10.0

# remove checkpoints for a benchmark run
rm /tigress/$USER/model_checkpoints/*
rm /tigress/$USER/results/*
rm /tigress/$USER/csv_logs/*
rm /tigress/$USER/Graph/*
rm /tigress/$USER/normalization/*

export OMPI_MCA_btl="tcp,self,vader"
srun python mpi_learn.py

# single model replica PyTorch implementation of Transformer
# (set one rank, one core, ...)
#
# conda activate Py3
# module load cudnn/cuda-10.0/7.5.0
# module load cudatoolkit/10.0
# module load openmpi/gcc/3.1.3/64
# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# srun python transformer_learn.py
