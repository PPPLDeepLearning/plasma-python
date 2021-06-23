#!/bin/bash
#SBATCH -C gpu
#SBATCH -t 01:30:00
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --exclusive

# rm /global/cscratch1/sd/$USER/model_checkpoints/*
# rm /global/cscratch1/sd/$USER/results/*
# rm /global/cscratch1/sd/$USER/csv_logs/*
# rm /global/cscratch1/sd/$USER/Graph/*
# rm /global/cscratch1/sd/$USER/normalization/*

export OMPI_MCA_btl="tcp,self,vader"
srun python mpi_learn.py
