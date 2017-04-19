#!/usr/bin/env python
import subprocess
from subprocess import Popen
from time import sleep

def checkAndSchedule(configBaseName,nextGPUcount,GPUstep,maxGPUcount):
    if nextGPUcount > maxGPUcount: return 
    job_is_running = subprocess.check_output(['qstat','-u','alexeys'])
    if len(job_is_running) > 0: 
        #sleep 500 seconds
        sleep(500)
        checkAndSchedule(configBaseName,nextGPUcount,GPUstep,maxGPUcount)
    else:
        #create a config
        nextConfigName = createOneConfig(configBaseName,nextGPUcount)
        print "Submitting next PBS job {} to run on {} GPUs".format(configBaseName,nextGPUcount)
        print "qsub "+nextConfigName
        Popen("qsub "+nextConfigName,shell=True).wait() 
        #update parameters
        nextGPUcount += GPUstep
        sleep(10)
        checkAndSchedule(configBaseName,nextGPUcount,GPUstep,maxGPUcount)


def createOneConfig(configBaseName, GPUcount):
    configFullName = configBaseName+str(GPUcount)+".cmd"
    with open(configFullName,"w") as f:
	f.write('#!/bin/bash\n')
	f.write('#PBS -A FUS117\n')
	f.write('#PBS -l walltime=1:30:00\n') #FIXME this depends a lot on the number of GPUs 1900s/1epoch at 50, 2350s/1epoch at 4
	f.write('#PBS -l nodes='+str(GPUcount)+'\n')
	f.write('##PBS -l procs=1\n')
	f.write('##PBS -l gres=atlas1%atlas2\n')
        f.write('\n\n')
	f.write('export HOME=/lustre/atlas/proj-shared/fus117\n')
	f.write('cd $HOME/PPPL/plasma-python/examples\n')
        f.write('\n\n')
	f.write('source $MODULESHOME/init/bash\n')
	f.write('module switch PrgEnv-pgi PrgEnv-gnu\n')
        f.write('\n\n')
	f.write('module load cudatoolkit\n')
	f.write('export LIBRARY_PATH=/opt/nvidia/cudatoolkit7.5/7.5.18-1.0502.10743.2.1/lib64:$LIBRARY_PATH\n')
        f.write('\n\n')
	f.write('#This block is CuDNN module\n')
	f.write('export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH\n')
	f.write('export LIBRARY_PATH=$HOME/cuda/lib64:$LIBRARY_PATH\n')
	f.write('export LDFLAGS=$LDFLAGS:$HOME/cuda/lib64\n')
	f.write('export INCLUDE=$INCLUDE:$HOME/cuda/include\n')
	f.write('export CPATH=$CPATH:$HOME/cuda/include\n')
	f.write('export FFLAGS=$FFLAGS:$HOME/cuda/include\n')
	f.write('export LOCAL_LDFLAGS=$LOCAL_LDFLAGS:$HOME/cuda/lib64\n')
	f.write('export LOCAL_INCLUDE=$LOCAL_INCLUDE:$HOME/cuda/include\n')
	f.write('export LOCAL_CFLAGS=$LOCAL_CFLAGS:$HOME/cuda/include\n')
	f.write('export LOCAL_FFLAGS=$LOCAL_FFLAGS:$HOME/cuda/include\n')
	f.write('export LOCAL_CXXFLAGS=$LOCAL_CXXFLAGS:$HOME/cuda/include\n')
        f.write('\n\n')
	f.write('#This sets new home and Anaconda module\n')
	f.write('export PATH=$HOME/anaconda2/bin:$PATH\n')
	f.write('export LD_LIBRARY_PATH=$HOME/anaconda2/lib:$LD_LIBRARY_PATH\n')
	f.write('source activate PPPL\n')
        f.write('\n\n')
	f.write('PYTHON=`which python`\n')
	f.write('echo $PYTHON\n')
        f.write('\n\n')
	f.write('export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH\n')
	f.write('export MPICH_RDMA_ENABLED_CUDA=1\n')
        f.write('\n\n')
	f.write('rm $HOME/tigress/alexeys/model_checkpoints/*\n')
	f.write('aprun -n'+str(GPUcount)+' -N1 $PYTHON mpi_learn.py\n')

    return configFullName

if __name__=='__main__':
    nextGPUcount = 50
    GPUstep = 50 
    maxGPUcount = 101
    configBaseName = "FRNN_Titan"
    checkAndSchedule(configBaseName,nextGPUcount,GPUstep,maxGPUcount)
