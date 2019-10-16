## Tutorials
*Last updated 2019-10-16.*

### Login to TigerGPU

First, login to TigerGPU cluster headnode via ssh:
```
ssh -XC <yourusername>@tigergpu.princeton.edu
```
Note, `-XC` is optional; it is only necessary if you are planning on performing remote visualization, e.g. the output `.png` files from the below [section](#Learning-curves-and-ROC-per-epoch). Trusted X11 forwarding can be used with `-Y` instead of `-X` and may prevent timeouts, but it disables X11 SECURITY extension controls. Compression `-C` reduces the bandwidth usage and may be useful on slow connections. 

### Sample usage on TigerGPU

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

#### Common issue

Common issue is Intel compiler mismatch in the `PATH` and what you use in the module. With the modules loaded as above,
you should see something like this:
```
$ which mpicc
/usr/local/openmpi/cuda-8.0/3.0.0/intel170/x86_64/bin/mpicc
```

If you `conda activate` the Anaconda environment **after** loading the OpenMPI library, your application would be built with the MPI library from Anaconda, which has worse performance on this cluster and could lead to errors. See [On Computing Well: Installing and Running ‘mpi4py’ on the Cluster](https://oncomputingwell.princeton.edu/2018/11/installing-and-running-mpi4py-on-the-cluster/) for a related discussion. 

#### Location of the data on Tigress

The JET and D3D datasets contain multi-modal time series of sensory measurements leading up to deleterious events called plasma disruptions. The datasets are located in the `/tigress/FRNN` project directory of the [GPFS](https://www.ibm.com/support/knowledgecenter/en/SSPT3X_3.0.0/com.ibm.swg.im.infosphere.biginsights.product.doc/doc/bi_gpfs_overview.html) filesystem on Princeton University clusters.

For convenience, create following symbolic links:
```bash
cd /tigress/<netid>
ln -s /tigress/FRNN/shot_lists shot_lists
ln -s /tigress/FRNN/signal_data signal_data
```

#### Preprocessing

```bash
cd examples/
python guarantee_preprocessed.py
```
This will preprocess the data and save rescaled copies of the signals in `/tigress/<netid>/processed_shots`, `/tigress/<netid>/processed_shotlists` and `/tigress/<netid>/normalization`

You would only have to run preprocessing once for each dataset. The dataset is specified in the config file `examples/conf.yaml`:
```yaml
paths:
    data: jet_data_0D
```    
Preprocessing this dataset takes about 20 minutes to preprocess in parallel and can normally be done on the cluster headnode.

#### Training and inference

Use Slurm scheduler to perform batch or interactive analysis on TigerGPU cluster.

##### Batch analysis

For batch analysis, make sure to allocate 1 MPI process per GPU. Save the following to `slurm.cmd` file (or make changes to the existing `examples/slurm.cmd`):

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

##### Interactive analysis

Interactive option is preferred for **debugging** or running in the **notebook**, for all other case batch is preferred.
The workflow is to request an interactive session:

```bash
salloc -N [X] --ntasks-per-node=4 --ntasks-per-socket=2 --gres=gpu:4 -c 4 --mem-per-cpu=0 -t 0-6:00
```
Then, launch the application from the command line:

```bash
mpirun -N 4 python mpi_learn.py
```
where `-N` is a synonym for `-npernode` in OpenMPI. Do **not** use `srun` to launch the job inside an interactive session. 

[//]: # (This option appears to be redundant given the salloc options; "mpirun python mpi_learn.py" appears to work just the same.)

[//]: # (HOWEVER, "srun python mpi_learn.py", "srun --ntasks-per-node python mpi_learn.py", etc. NEVER works--- it just hangs without any output. Why?)

[//]: # (Consistent with https://www.open-mpi.org/faq/?category=slurm ?)

[//]: # (certain output seems to be repeated by ntasks-per-node, e.g. echoing the conf.yaml. Expected?)


### Understanding the data

All the configuration parameters are summarised in `examples/conf.yaml`. Highlighting the important ones to control the data.
Currently, FRNN is capable of working with JET and D3D data as well as cross-machine regime. The switch is done in the configuration file:

```yaml
paths:
    ... 
    data: 'jet_data_0D'
```
use `d3d_data` for D3D signals, use `jet_to_d3d_data` ir `d3d_to_jet_data` for cross-machine regime.
    
By default, FRNN will select, preprocess and normalize all valid signals available. To chose only specific signals use:
```yaml
paths:
    ... 
    specific_signals: [q95,ip] 
```    
if left empty `[]` will use all valid signals defined on a machine. Only use if need a custom set.

Other parameters configured in the conf.yaml include batch size, learning rate, neural network topology and special conditions foir hyperparameter sweeps.

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

### Visualizing learning

A regular FRNN run will produce several outputs and callbacks.

#### TensorBoard visualization

Currently supports graph visualization, histograms of weights, activations and biases, and scalar variable summaries of losses and accuracies.

The summaries are written in real time to `/tigress/<netid>/Graph`. For macOS, you can set up the `sshfs` mount of the `/tigress` filesystem and view those summaries in your browser.

To install SSHFS on a macOS system, you could follow the instructions here:
https://github.com/osxfuse/osxfuse/wiki/SSHFS
Or use [Homebrew](https://brew.sh/), `brew cask install osxfuse; brew install sshfs`. Note, to install and/or use `osxfuse` you may need to enable its kernel extension in: System Preferences → Security & Privacy → General

then do something like:
```
sshfs -o allow_other,defer_permissions netid@tigergpu.princeton.edu:/tigress/<netid>/ <destination folder name on your laptop>/
```

Launch TensorBoard locally (assuming that it is installed on your local computer):
```
python -m tensorboard.main --logdir <destination folder name on your laptop>/Graph
```
A URL should be emitted to the console output. Navigate to this link in your browser. If the TensorBoard interface does not open, try directing your browser to `localhost:6006`.

You should see something like:

![tensorboard example](https://github.com/PPPLDeepLearning/plasma-python/blob/master/docs/tb.png)

#### Learning curves and ROC per epoch

Besides TensorBoard summaries you can produce the ROC curves for validation and test data as well as visualizations of shots:
```
cd examples/
python performance_analysis.py
```
this uses the resulting file produced as a result of training the neural network as an input, and produces several `.png` files with plots as an output.

In addition, you can check the scalar variable summaries for training loss, validation loss and validation ROC logged at `/tigress/<netid>/csv_logs` (each run will produce a new log file with a timestamp in name).

A sample code to analyze can be found in `examples/notebooks`. For instance:

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

### Learning curve summaries per mini-batch

To extract per mini-batch summaries, use the output produced by FRNN logged to the standard out (in case of the batch jobs, it will all be contained in the Slurm output file). Refer to the following notebook to perform the analysis of learning curve on a mini-batch level: [FRNN_scaling.ipynb](https://github.com/PPPLDeepLearning/plasma-python/blob/master/examples/notebooks/FRNN_scaling.ipynb)
