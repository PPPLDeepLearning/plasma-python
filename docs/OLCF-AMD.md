# OLCF Spock  Tutorial
*Last updated 2021-8-19*

*This document is built off of the excellent how-to guide created for [Princeton's TigerGPU](https://github.com/Techercise/plasma-python/blob/master/docs/PrincetonUTutorial.md)*

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
In order to load the correct modules with ease, creating a profile is recommended
```
vim frnn_spock.profile
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
vim mpi.cfg
```

Include the following segment in the mpi.cfg file:
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
### Location of the data on Spock

**Currently, no public data exists on Spock, but we leave this section in here for the user to understand the input data**

The JET and D3D datasets contain multi-modal time series of sensory measurements leading up to deleterious events called plasma disruptions. The datasets are located in the `/tigress/FRNN` project directory of the [GPFS](https://www.ibm.com/support/knowledgecenter/en/SSPT3X_3.0.0/com.ibm.swg.im.infosphere.biginsights.product.doc/doc/bi_gpfs_overview.html) filesystem on Princeton University clusters.

For convenience, create following symbolic links:
```bash
cd /tigress/<netid>
ln -s /tigress/FRNN/shot_lists shot_lists
ln -s /tigress/FRNN/signal_data signal_data
```

### Configuring the dataset
All the configuration parameters are summarised in `examples/conf.yaml`. In this section, we highlight the important ones used to control the input data. 

Currently, FRNN is capable of working with JET and D3D data as well as thecross-machine regime. The switch is done in the configuration file:
```yaml
paths:
    ... 
    data: 'jet_0D'
```

Older yaml files kept for archival purposes will denote this data set as follow:
```yaml
paths:
    ... 
    data: 'jet_data_0D'
```
use `d3d_data` for D3D signals, use `jet_to_d3d_data` ir `d3d_to_jet_data` for cross-machine regime.
    
By default, FRNN will select, preprocess, and normalize all valid signals available in the above dataset. To chose only specific signals use:
```yaml
paths:
    ... 
    specific_signals: [q95,ip] 
```    
if left empty `[]` will use all valid signals defined on a machine. Only set this variable if you need a custom set of signals.

Other parameters configured in the `conf.yaml` include batch size, learning rate, neural network topology and special conditions foir hyperparameter sweeps.

### Preprocessing the input data
***Preprocessing the input data is currently not required on Spock as the data that is available is already preprocessed.***

```bash
cd examples/
python guarantee_preprocessed.py
```
This will preprocess the data and save rescaled copies of the signals in `/tigress/<netid>/processed_shots`, `/tigress/<netid>/processed_shotlists` and `/tigress/<netid>/normalization`

Preprocessing must be performed only once per each dataset. For example, consider the following dataset specified in the config file `examples/conf.yaml`:
```yaml
paths:
    data: jet_0D
```    
Preprocessing this dataset takes about 20 minutes to preprocess in parallel and can normally be done on the cluster headnode.

### Current signals and notations

Signal name | Description 
--- | --- 
q95 | q95 safety factor
ip | plasma current
li | internal inductance 
lm | Locked mode amplitude
dens | Plasma density
energy | stored energy
pin | Input Power (beam for d3d)
pradtot | Radiated Power
pradcore | Radiated Power Core
pradedge | Radiated Power Edge
pechin | ECH input power, not always on
pechin | ECH input power, not always on
betan | Normalized Beta
energydt | stored energy time derivative
torquein | Input Beam Torque
tmamp1 | Tearing Mode amplitude (rotating 2/1)
tmamp2 | Tearing Mode amplitude (rotating 3/2)
tmfreq1 | Tearing Mode frequency (rotating 2/1)
tmfreq2 | Tearing Mode frequency (rotating 3/2)
ipdirect | plasma current direction

## Training and inference

Use the Slurm job scheduler to perform batch or interactive analysis on the Spock system.

### Batch job

A sample batch job script for 1 GPU is provided in the examples directory and is called spock_1GPU_slurm.cmd. It can be run using: `sbatch spock_1GPU_slurm.cmd`
Note that, the project/account (`-A`) and partition (`-p) arugments will need to reflect your project and assigned partition.

Some batch job tips:
* For non-interactive batch analysis, make sure to allocate exactly 1 MPI process per GPU where `X` is the number of nodes for distibuted training and the total number of GPUs is `X * 4`. This configuration guarantees 1 MPI process per GPU, regardless of the value of `X`. 
* Update the `num_gpus` value in `conf.yaml` to correspond to the total number of GPUs specified for your Slurm allocation.

And monitor it's completion via:
```bash
squeue --me
```
Optionally, add an email notification option in the Slurm configuration about the job completion:
```
#SBATCH --mail-user=<userid>@email.com
#SBATCH --mail-type=ALL
```

### Interactive job

Interactive option is preferred for **debugging** or running in the **notebook**, for all other case batch is preferred.
The workflow is to request an interactive session for a 1 GPU interactive job:

```bash
salloc -t 02:00:00 -A <project_id> -N 1 --gres=gpu:1 --exclusive -p <partition> --ntasks-per-socket=1 --ntasks-per-node=1
```

[//]: # (Note, the modules might not/are not inherited from the shell that spawns the interactive Slurm session. Need to reload anaconda module, activate environment, and reload other compiler/library modules)

Ensure the above modules are still loaded and reactivate your conda environmnt.
Then, launch the application from the command line:

```bash
python mpi_learn.py
```

## Visualizing learning

A regular FRNN run will produce several outputs and callbacks.

## Custom visualization
You can visualize the accuracy of the trained FRNN model using the custom Python scripts and notebooks included in the repository.

### Learning curves, example shots, and ROC per epoch

You can produce the ROC curves for validation and test data as well as visualizations of shots by using:
```
cd examples/
python performance_analysis.py
```
The `performance_analysis.py` script uses the file produced as a result of training the neural network as an input, and produces several `.png` files with plots as an output.

In addition, you can check the scalar variable summaries for training loss, validation loss, and validation ROC logged at `/outputdir/<userid>/csv_logs` (each run will produce a new log file with a timestamp in name).

Sample notebooks for analyzing the files in this directory can be found in `examples/notebooks/`. For instance, the [LearningCurves.ipynb](https://github.com/PPPLDeepLearning/plasma-python/blob/master/examples/notebooks/LearningCurves.ipynb) notebook contains a variation on the following code snippet:
```python
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file, save

data = pd.read_csv("<destination folder name on your laptop>/csv_logs/<name of the log file>.csv")

from bokeh.io import output_notebook
output_notebook()

from bokeh.models import Range1d
#optionally set the plotting range
#left, right, bottom, top = -0.1, 31, 0.005, 1.51

p = figure(title="Learning curve", y_axis_label="Training loss", x_axis_label='Epoch number') #,y_axis_type="log")
#p.set(x_range=Range1d(left, right), y_range=Range1d(bottom, top))

p.line(data['epoch'].values, data['train_loss'].values, legend="Test description",
       line_color="tomato", line_dash="dotdash", line_width=2)
p.legend.location = "top_right"
show(p, notebook_handle=True)
```
The resulting plot should match the `train_loss` plot in the Scalars tab of the TensorBoard summary. 

#### Learning curve summaries per mini-batch

To extract per mini-batch summaries, we require a finer granularity of checkpoint data than what it is logged to the per-epoch lines of `csv_logs/` files. We must directly use the output produced by FRNN logged to the standard output stream. In the case of the non-interactive Slurm batch jobs, it will all be contained in the Slurm output file, e.g. `slurm-3842170.out`. Refer to the following notebook to perform the analysis of learning curve on a mini-batch level: [FRNN_scaling.ipynb](https://github.com/PPPLDeepLearning/plasma-python/blob/master/examples/notebooks/FRNN_scaling.ipynb)
