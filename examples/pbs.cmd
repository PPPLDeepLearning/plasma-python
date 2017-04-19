#!/bin/bash
#PBS -A FUS117
#PBS -l walltime=0:05:00
#PBS -l nodes=2
##PBS -l procs=1
##PBS -l gres=atlas1%atlas2


#Cannot see home folder, will just hang until wall limit
export HOME=/ccs/proj/fus117/
cd $HOME/PPPL/plasma-python/examples

source $MODULESHOME/init/bash
module switch PrgEnv-pgi PrgEnv-gnu

module load cudatoolkit
export LIBRARY_PATH=/opt/nvidia/cudatoolkit7.5/7.5.18-1.0502.10743.2.1/lib64:$LIBRARY_PATH

#This block is CuDNN module
export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/cuda/lib64:$LIBRARY_PATH
export LDFLAGS=$LDFLAGS:$HOME/cuda/lib64
export INCLUDE=$INCLUDE:$HOME/cuda/include
export CPATH=$CPATH:$HOME/cuda/include
export FFLAGS=$FFLAGS:$HOME/cuda/include
export LOCAL_LDFLAGS=$LOCAL_LDFLAGS:$HOME/cuda/lib64
export LOCAL_INCLUDE=$LOCAL_INCLUDE:$HOME/cuda/include
export LOCAL_CFLAGS=$LOCAL_CFLAGS:$HOME/cuda/include
export LOCAL_FFLAGS=$LOCAL_FFLAGS:$HOME/cuda/include
export LOCAL_CXXFLAGS=$LOCAL_CXXFLAGS:$HOME/cuda/include

#This sets new home and Anaconda module
export PATH=$HOME/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=$HOME/anaconda2/lib:$LD_LIBRARY_PATH
source activate PPPL

PYTHON=`which python`
echo $PYTHON

#pygpu backend
#export CPATH=$CPATH:~/.local/include
#export LIBRARY_PATH=$LIBRARY_PATH:~/.local/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export MPICH_RDMA_ENABLED_CUDA=1

rm $HOME/tigress/alexeys/model_checkpoints/*
aprun -n2 -N1 $PYTHON mpi_learn.py
