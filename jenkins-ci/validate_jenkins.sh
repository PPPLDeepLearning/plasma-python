#!/bin/bash

export OMPI_MCA_btl="tcp,self,sm"

module load anaconda3/4.4.0
source activate /tigress/alexeys/jenkins/.conda/envs/jenkins3
module load cudatoolkit/8.0
module load cudnn/cuda-8.0/6.0
module load openmpi/cuda-8.0/intel-17.0/2.1.0/64
module load intel/17.0/64/17.0.4.196

cd /home/alexeys/jenkins/workspace/FRNM/PPPL
python setup.py install

echo `which python`
echo `which mpicc`

echo ${PWD}
echo $SLURM_NODELIST

cd jenkins-ci
echo ${PWD}
ls ${PWD}

srun python validate_jenkins.py
