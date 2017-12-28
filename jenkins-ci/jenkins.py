#!/bin/bash
from subprocess import Popen

Popen("export OMPI_MCA_btl=\"tcp,self,sm\"",shell=True)

Popen("echo \"Jenkins test Python3.6\"",shell=True)
Popen("rm /tigress/alexeys/model_checkpoints/*",shell=True)
Popen("rm -rf /tigress/alexeys/processed_shots",shell=True)
Popen("rm -rf /tigress/alexeys/processed_shotlists",shell=True)
Popen("rm -rf /tigress/alexeys/normalization",shell=True)
Popen("module load anaconda3",shell=True)
Popen("source activate PPPL_dev3",shell=True)
Popen("module load cudatoolkit/8.0",shell=True)
Popen("module load cudnn/cuda-8.0/6.0",shell=True)
Popen("module load openmpi/cuda-8.0/intel-17.0/2.1.0/64",shell=True)
Popen("module load intel/17.0/64/17.0.4.196",shell=True)

Popen("python setup.py install",shell=True)

Popen("sed -i -e 's/num_epochs: 1000/num_epochs: 2/g' conf.yaml",shell=True)
Popen("sed -i -e 's/data: jet_data/data: jenkins_jet/g' conf.yaml",shell=True)

Popen("echo $SLURM_NODELIST",shell=True)
Popen("cd examples",shell=True)
Popen("srun python mpi_learn.py",shell=True).wait()

Popen("echo \"Jenkins test Python2.7\"",shell=True)
Popen("rm /tigress/alexeys/model_checkpoints/*",shell=True)
Popen("rm -rf /tigress/alexeys/processed_shots",shell=True)
Popen("rm -rf /tigress/alexeys/processed_shotlists",shell=True)
Popen("rm -rf /tigress/alexeys/normalization",shell=True)

Popen("source deactivate",shell=True)
Popen("module purge",shell=True)
Popen("module load anaconda",shell=True)
Popen("source activate PPPL",shell=True)
Popen("module load cudatoolkit/8.0",shell=True)
Popen("module load cudnn/cuda-8.0/6.0",shell=True)
Popen("module load openmpi/cuda-8.0/intel-17.0/2.1.0/64",shell=True)
Popen("module load intel/17.0/64/17.0.4.196",shell=True)

Popen("cd ..",shell=True)
Popen("python setup.py install",shell=True)

Popen("sed -i -e 's/data: jenkins_jet/data: jenkins_d3d/g' conf.yaml",shell=True)

Popen("echo $SLURM_NODELIST",shell=True)
Popen("cd examples",shell=True)
Popen("srun python mpi_learn.py",shell=True).wait()
