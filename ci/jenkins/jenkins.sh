#!/bin/bash

export OMPI_MCA_btl="tcp,self,sm"

echo ${PWD}

echo "Jenkins test Python3.6"
rm /tigress/alexeys/model_checkpoints/*
rm -rf /tigress/alexeys/processed_shots
rm -rf /tigress/alexeys/processed_shotlists
rm -rf /tigress/alexeys/normalization
module load anaconda3/4.4.0
source activate /tigress/alexeys/jenkins/.conda/envs/jenkins3
module load cudatoolkit/8.0
module load cudnn/cuda-8.0/6.0
module load openmpi/cuda-8.0/intel-17.0/2.1.0/64
module load intel/17.0/64/17.0.4.196

echo ${PWD}
cd /home/alexeys/jenkins/workspace/FRNM/PPPL
echo ${PWD}
python setup.py install

echo $SLURM_NODELIST
cd examples
echo ${PWD}
ls ${PWD}
sed -i -e 's/num_epochs: 1000/num_epochs: 2/g' conf.yaml
sed -i -e 's/data: jet_data/data: jenkins_jet/g' conf.yaml

srun python mpi_learn.py

echo "Jenkins test Python2.7"
