#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH --mem-per-cpu=0

module load anaconda/4.4.0
source activate PPPL
module load cudatoolkit/8.0
module load cudnn/cuda-8.0/6.0
module load openmpi/cuda-8.0/intel-17.0/2.1.0/64
module load intel/17.0/64/17.0.4.196

#remove checkpoints for a benchmark run
rm /scratch/gpfs/$USER/model_checkpoints/*
rm /scratch/gpfs/$USER/results/*
rm /scratch/gpfs/$USER/csv_logs/*
rm /scratch/gpfs/$USER/Graph/*
rm /scratch/gpfs/$USER/normalization/*

export OMPI_MCA_btl="tcp,self,sm"
srun python mpi_learn.py
