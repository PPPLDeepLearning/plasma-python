# TigerGPU Tutorial
*Last updated 2019-10-24.*

## Building the package
### Login to TigerGPU

First, login to TigerGPU cluster headnode via ssh:
```
ssh -XC <yourusername>@tigergpu.princeton.edu
```
Note, `-XC` is optional; it is only necessary if you are planning on performing remote visualization, e.g. the output `.png` files from the below [section](#Learning-curves-and-ROC-per-epoch). Trusted X11 forwarding can be used with `-Y` instead of `-X` and may prevent timeouts, but it disables X11 SECURITY extension controls. Compression `-C` reduces the bandwidth usage and may be useful on slow connections. 

### Sample installation on TigerGPU

Next, check out the source code from github:
```
git clone https://github.com/PPPLDeepLearning/plasma-python
cd plasma-python
```

After that, create an isolated Anaconda environment and load CUDA drivers, an MPI compiler, and the HDF5 library:
```
#cd plasma-python
module load anaconda3
conda create --name my_env --file requirements-travis.txt
conda activate my_env

export OMPI_MCA_btl="tcp,self,vader"
# replace "vader" with "sm" for OpenMPI versions prior to 3.0.0
module load cudatoolkit cudann 
module load openmpi/cuda-8.0/intel-17.0/3.0.0/64
module load intel
module load hdf5/intel-17.0/intel-mpi/1.10.0
```
As of the latest update of this document, the above modules correspond to the following versions on the TigerGPU system, given by `module list`:
```
Currently Loaded Modulefiles:
  1) anaconda3/2019.3                       4) openmpi/cuda-8.0/intel-17.0/3.0.0/64   7) hdf5/intel-17.0/intel-mpi/1.10.0
  2) cudatoolkit/10.1                       5) intel-mkl/2019.3/3/64
  3) cudnn/cuda-9.2/7.6.3                   6) intel/19.0/64/19.0.3.199
```

Next, install the `plasma-python` package:

```bash
#conda activate my_env
python setup.py install
```

Where `my_env` should contain the Python packages as per `requirements-travis.txt` file.

### Common build issue: cluster's MPI library and `mpi4py`

Common issue is Intel compiler mismatch in the `PATH` and what you use in the module. With the modules loaded as above,
you should see something like this:
```
$ which mpicc
/usr/local/openmpi/cuda-8.0/3.0.0/intel170/x86_64/bin/mpicc
```
Especially note the presence of the CUDA directory in this path. This indicates that the loaded OpenMPI library is [CUDA-aware](https://www.open-mpi.org/faq/?category=runcuda).

If you `conda activate` the Anaconda environment **after** loading the OpenMPI library, your application would be built with the MPI library from Anaconda, which has worse performance on this cluster and could lead to errors. See [On Computing Well: Installing and Running ‘mpi4py’ on the Cluster](https://oncomputingwell.princeton.edu/2018/11/installing-and-running-mpi4py-on-the-cluster/) for a related discussion. 


## Understanding and preparing the input data
### Location of the data on Tigress

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

```bash
cd examples/
python guarantee_preprocessed.py
```
This will preprocess the data and save rescaled copies of the signals in `/tigress/<netid>/processed_shots`, `/tigress/<netid>/processed_shotlists` and `/tigress/<netid>/normalization`

Preprocessing must be performed only once per each dataset. For example, consider the following dataset specified in the config file `examples/conf.yaml`:
```yaml
paths:
    data: jet_data_0D
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

Use the Slurm job scheduler to perform batch or interactive analysis on TigerGPU cluster.

### Batch job

For non-interactive batch analysis, make sure to allocate exactly 1 MPI process per GPU. Save the following to `slurm.cmd` file (or make changes to the existing `examples/slurm.cmd`):

```bash
#!/bin/bash
#SBATCH -t 01:30:00
#SBATCH -N X
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH --mem-per-cpu=0

module load anaconda3
conda activate my_env
export OMPI_MCA_btl="tcp,self,vader"
module load cudatoolkit cudann 
module load openmpi/cuda-8.0/intel-17.0/3.0.0/64
module load intel
module load hdf5/intel-17.0/intel-mpi/1.10.0

srun python mpi_learn.py

```
where `X` is the number of nodes for distibuted training and the total number of GPUs is `X * 4`. This configuration guarantees 1 MPI process per GPU, regardless of the value of `X`. 

Update the `num_gpus` value in `conf.yaml` to correspond to the total number of GPUs specified for your Slurm allocation.

Submit the job with (assuming you are still in the `examples/` subdirectory):
```bash
#cd examples
sbatch slurm.cmd
```

And monitor it's completion via:
```bash
squeue -u <netid>
```
Optionally, add an email notification option in the Slurm configuration about the job completion:
```
#SBATCH --mail-user=<netid>@princeton.edu
#SBATCH --mail-type=ALL
```

### Interactive job

Interactive option is preferred for **debugging** or running in the **notebook**, for all other case batch is preferred.
The workflow is to request an interactive session:

```bash
salloc -N [X] --ntasks-per-node=4 --ntasks-per-socket=2 --gres=gpu:4 -c 4 --mem-per-cpu=0 -t 0-6:00
```

[//]: # (Note, the modules might not/are not inherited from the shell that spawns the interactive Slurm session. Need to reload anaconda module, activate environment, and reload other compiler/library modules)

Re-load the above modules and reactivate your conda environment. Confirm that the correct CUDA-aware OpenMPI library is in your interactive Slurm sessions's shell search path:
```bash
$ which mpirun 
/usr/local/openmpi/cuda-8.0/3.0.0/intel170/x86_64/bin/mpirun
```
Then, launch the application from the command line:

```bash
mpirun -N 4 python mpi_learn.py
```
where `-N` is a synonym for `-npernode` in OpenMPI. Do **not** use `srun` to launch the job inside an interactive session. If 
you an encounter an error such as "unrecognized argument N", it is likely that your modules are incorrect and point to an Intel MPI distribution instead of CUDA-aware OpenMPI. Intel MPI is based on MPICH, which does not offer the `-npernode` option. You can confirm this by checking:
```bash
$ which mpirun 
/opt/intel/compilers_and_libraries_2019.3.199/linux/mpi/intel64/bin/mpirun
```

[//]: # (This option appears to be redundant given the salloc options; "mpirun python mpi_learn.py" appears to work just the same.)

[//]: # (HOWEVER, "srun python mpi_learn.py", "srun --ntasks-per-node python mpi_learn.py", etc. NEVER works--- it just hangs without any output. Why? ANSWER: salloc starts a session on the node using srun under the covers, which may consume a GPU in the allocation. Next srun call will hang due to a lack of required resources. Wrapper fix to this may have been extended from mpirun to srun on 2019-10-22)

## Visualizing learning

A regular FRNN run will produce several outputs and callbacks.

### TensorBoard visualization

Currently supports graph visualization, histograms of weights, activations and biases, and scalar variable summaries of losses and accuracies.

The summaries are written in real time to `/tigress/<netid>/Graph`. For macOS, you can set up the `sshfs` mount of the [`/tigress`](https://researchcomputing.princeton.edu/storage/tigress) filesystem and view those summaries in your browser.

To install SSHFS on a macOS system, you could follow the instructions here:
https://github.com/osxfuse/osxfuse/wiki/SSHFS
Or use [Homebrew](https://brew.sh/), `brew cask install osxfuse; brew install sshfs`. Note, to install and/or use `osxfuse` you may need to enable its kernel extension in: System Preferences → Security & Privacy → General

After installation, execute:
```
sshfs -o allow_other,defer_permissions netid@tigergpu.princeton.edu:/tigress/<netid>/ <destination folder name on your laptop>/
```
The local destination folder may be an existing (possibly nonempty) folder. If it does not exist, SSHFS will create the folder. You can confirm that the operation succeeded via the `mount` command, which prints the list of currently mounted filesystems if no arguments are given.

Launch TensorBoard locally (assuming that it is installed on your local computer):
```
python -m tensorboard.main --logdir <destination folder name on your laptop>/Graph
```
A URL should be emitted to the console output. Navigate to this link in your browser. If the TensorBoard interface does not open, try directing your browser to `localhost:6006`.

You should see something like:

![tensorboard example](https://github.com/PPPLDeepLearning/plasma-python/blob/master/docs/images/tb.png)

When you are finished with analyzing the summaries in TensorBoard, you may wish to unmount the remote filesystem:
```
umount  <destination folder name on your laptop>
```
The local destination folder will remain present, but it will no longer contain the remote files. It will be returned to its previous state, either empty or containing the original local files. Note, the `umount` command is appropriate for macOS systems; some Linux systems instead offer the `fusermount` command.  

These commands may be useful when the SSH connection is lost and an existing mount point cannot be re-mounted, e.g. errors such as:
```
mount_osxfuse: mount point <destination folder name on your laptop> is itself on a OSXFUSE volume
```

More aggressive options such as `umount -f <destination folder name on your laptop>` and alternative approaches may be necessary; see [discussion here](https://github.com/osxfuse/osxfuse/issues/45#issuecomment-21943107).

## Custom visualization
Besides TensorBoard summaries, you can visualize the accuracy of the trained FRNN model using the custom Python scripts and notebooks included in the repository.

### Learning curves, example shots, and ROC per epoch

You can produce the ROC curves for validation and test data as well as visualizations of shots by using:
```
cd examples/
python performance_analysis.py
```
The `performance_analysis.py` script uses the file produced as a result of training the neural network as an input, and produces several `.png` files with plots as an output.

[//]: # (Add details about sig_161308test.npz, disruptive_alarms_test.npz, 4x metric* png, accum_disruptions.png, test_roc.npz)

In addition, you can check the scalar variable summaries for training loss, validation loss, and validation ROC logged at `/tigress/<netid>/csv_logs` (each run will produce a new log file with a timestamp in name).

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
