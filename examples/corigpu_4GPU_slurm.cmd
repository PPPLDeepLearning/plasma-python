#!/bin/bash
#SBATCH -C gpu
#SBATCH -t 02:00:00
#SBATCH -N 4
#SBATCH -G 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH -c 4
#SBATCH --mem-per-cpu=0
#SBATCH --exclusive

# rm /global/cscratch1/sd/$USER/model_checkpoints/*
# rm /global/cscratch1/sd/$USER/results/*
# rm /global/cscratch1/sd/$USER/csv_logs/*
# rm /global/cscratch1/sd/$USER/Graph/*
# rm /global/cscratch1/sd/$USER/normalization/*

export OMPI_MCA_btl="tcp,self,vader"
srun python mpi_learn.py
