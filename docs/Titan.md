# Set home at PROJWORK in the .bashrc:
```bash
export HOME=/lustre/atlas/proj-shared/fus117/
```

#cd ~


#Set up CUDA:
```bash
wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-7.5-linux-x64-v5.1.tgz
tar -xvf 
```

Add following to the submission script:

```bash
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
```


Add LIBRARY_PATH in addition to cudatoolkit:
```bash
module load cudatoolkit
export LIBRARY_PATH=/opt/nvidia/cudatoolkit7.5/7.5.18-1.0502.10743.2.1/lib64:$LIBRARY_PATH
```

# Download and install Anaconda
wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
sh A..


do not add PATH to .bashrc - it messes up modules for some reason


# Clone the PPPL repo

add ssh keys to github to ~/.ssh
ssh-add ~/.ssh/olcf_github_rsa

git clone git@github.com:PPPLDeepLearning/plasma-python.git
cd PPPL/plasma-python

Create PPPL env:
conda create --name PPPL --file requirements.txt

#Install mpi4py

module switch PrgEnv-pgi PrgEnv-gnu
export MPICC=cc
python setup.py install


doing custom installs with pip --user is OK


#Make sure to update paths in the conf.yaml


#The mass batch job submission is performed with this script:
https://github.com/PPPLDeepLearning/plasma-python/blob/titan_setup/examples/prepare_pbs_configs_titan.py
