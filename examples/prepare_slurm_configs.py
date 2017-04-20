#!/usr/bin/env python
import subprocess
from subprocess import Popen
from time import sleep

def checkAndSchedule(configBaseName,gpuNodeCountGrid,nextGPUNodeCount):
    if nextGPUNodeCount > len(gpuNodeCountGrid)-1: return 
    job_is_running = subprocess.check_output(['squeue','-u','alexeys']) #['qstat','-u','alexeys'])
    if 'alexeys' in job_is_running:
        #sleep 500 seconds
        sleep(500)
        checkAndSchedule(configBaseName,gpuNodeCountGrid,nextGPUNodeCount)
    else:
        #create a config
        nextConfigName = createOneConfig(configBaseName,gpuNodeCountGrid[nextGPUNodeCount])
        print "Submitting next PBS job {} to run on {} GPUs".format(configBaseName,gpuNodeCountGrid[nextGPUNodeCount])
        print "sbatch "+nextConfigName
        Popen("sbatch "+nextConfigName,shell=True).wait() 
        #update parameters
        nextGPUNodeCount += 1
        sleep(10)
        checkAndSchedule(configBaseName,gpuNodeCountGrid,nextGPUNodeCount)


def createOneConfig(configBaseName, GPUcount):
    configFullName = configBaseName+str(GPUcount)+".cmd"
    with open(configFullName,"w") as f:
	f.write('#!/bin/bash\n')
        f.write('#SBATCH -t 01:00:00\n')
        f.write('#SBATCH -N '+str(GPUcount)+'\n')
        f.write('#SBATCH --ntasks-per-node=4\n')
        f.write('#SBATCH --ntasks-per-socket=2\n')
        f.write('#SBATCH --gres=gpu:4\n')
        f.write('#SBATCH -c 4\n')
        f.write('\n\n')
        f.write('module load anaconda\n')
        f.write('source activate PPPL\n')
        f.write('module load cudatoolkit/8.0 cudann/cuda-8.0/5.1 openmpi/intel-17.0/1.10.2/64 intel/17.0/64/17.0.2.174\n')

        f.write('rm /tigress/alexeys/model_checkpoints/*\n')
        f.write('srun python mpi_learn.py\n')	

    return configFullName

if __name__=='__main__':
    gpuNodeCountGrid = [1,3,6,12,24,32,48]
    configBaseName = "FRNN_TigerGPU"
    checkAndSchedule(configBaseName,gpuNodeCountGrid,0)
