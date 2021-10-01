# OLCF Spock  Tutorial
*Last updated 2021-10-1*

*This document is built off of the excellent how-to guide created for [Princeton's TigerGPU](./PrincetonUTutorial.md)*

## Building the package
### Login to Spock

First, login to the Spock headnode via ssh:
```
ssh -X <yourusername>@spock.olcf.ornl.gov
```
Note, `-X` is optional; it is only necessary if you are planning on performing remote visualization, e.g. the output `.png` files from the below [section](#Learning-curves-and-ROC-per-epoch). Trusted X11 forwarding can be used with `-Y` instead of `-X` and may prevent timeouts, but it disables X11 SECURITY extension controls. 

### Sample installation on Spock

#### Check out the Code Repository
Next, check out the source code from github:
```
git clone https://github.com/PPPLDeepLearning/plasma-python
cd plasma-python
```

#### Install Miniconda
At the time of writing, Anaconda and Miniconda are not installed on Spock, therefore one of them must be manually downloaded. In their system documentation, AMD recommends downloading Miniconda. 

To install Miniconda, download the Linux installer [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers) and follow the installation instructions for Miniconda on [this page](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Once Miniconda is installed, create a conda environment:
```
conda create -n your_env_name python=3.8 -y
```

Then, activate the environment:
```
conda activate your_env_name
```

Ensure the following packages are installed in your conda environment:
```
pyyaml            # pip install pyyaml
pathos            # pip install pathos
hyperopt          # pip install hyperopt
matplotlib        # pip install matplotlib
keras             # pip install keras
tensorflow-rocm   # pip install tensorflow-rocm
```

#### Modules
In order to load the correct modules with ease, creating a profile is recommended. Create a profile named
```
frnn_spock.profile
```

Write the following to the profile:
```
module load rocm
module load cray-python
module load gcc
module load craype-accel-amd-gfx908
module load cray-mpich/8.1.7
module use /sw/aaims/spock/modulefiles
module load tensorflow

# These must be set before running if wanting to use the Cray GPU-Aware MPI
# If running on only 1 GPU, there is no need to uncomment these lines

# export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
# export MPICH_GPU_SUPPORT_ENABLED=1
# export HIPCC_COMPILE_FLAGS_APPEND="$HIPCC_COMPILE_FLAGS_APPEND -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi -L/opt/cray/pe/mpich/8.1.7/gtl/lib -lmpi_gtl_hsa"

export MPICC="$(which mpicc)"
```


As of the latest update of this document (Summer 2021), the above modules correspond to the following versions on the Spock system, given by `module list` (Note that this list also includes the default system modules):
```
Currently Loaded Modules:
  1) craype/2.7.8      3) libfabric/1.11.0.4.75   5) cray-dsmml/0.1.5         7) xpmem/2.2.40-2.1_2.28__g3cf3325.shasta   9) cray-pmi/6.0.12      11) DefApps/default    13) cray-python/3.8.5.1  15) craype-accel-amd-gfx908  17) rocm/4.1.0
  2) craype-x86-rome   4) craype-network-ofi      6) perftools-base/21.05.0   8) cray-libsci/21.06.1.1                   10) cray-pmi-lib/6.0.12  12) PrgEnv-cray/8.1.0  14) gcc/10.3.0           16) cray-mpich/8.1.7         18) tensorflow/2.3.6
```

#### Build mpi4py
If wanting to run on multiple GPUs, mpi4py is needed. At the time of writing, a manual installation of mpi4py is needed on the Spock system. To install mpi4py, do the following:
```
# Ensure your conda environment is activated:
conda activate your_env_name

# Download mpi4py to your home directory
#cd ~
curl -O -L https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.3.tar.gz

# Untar the file
tar -xzvf mpi4py-3.0.3.tar.gz

cd mpi4py-3.0.3

# Edit the mpi.cfg file
mpi.cfg
```

Include the following segment in the `mpi.cfg` file:
```
      [craympi]
      mpi_dir              = /opt/cray/pe/mpich/8.1.4/ofi/crayclang/9.1
      mpicc                = cc
      mpicxx               = CC
      include_dirs         = /opt/cray/pe/mpich/8.1.4/ofi/crayclang/9.1/include
      libraries            = mpi
      library_dirs         = /opt/cray/pe/mpich/8.1.4/ofi/crayclang/9.1/
```

Build and install mpi4py:
```
python setup.py build --mpi=craympi
python setup.py install
```

Next, install the `plasma-python` package:

```bash
#conda activate your_env_name
#cd ~/plasma-python
python setup.py install
```

## Understanding and preparing the input data

To learn how to understand and prepare the input data, please see the [corresponding section in the TigerGPU tutorial](./PrincetonUTutorial.md#understanding-and-preparing-the-input-data)

