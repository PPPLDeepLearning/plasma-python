## Tutorials

### Sample usage on Tigergpu

First, create an isolated Anaconda environment and load CUDA drivers:
```
module load anaconda3
module load cudatoolkit/8.0 cudnn/cuda-8.0/6.0 openmpi/cuda-8.0/intel-17.0/2.1.0/64 intel/17.0/64/17.0.2.174
module load intel/17.0/64/17.0.4.196 intel-mkl/2017.3/4/64
conda create --name my_env --file requirements-travis.txt
source activate my_env
```

Then install the plasma-python package:

```bash
#source activate my_env
git clone https://github.com/PPPLDeepLearning/plasma-python
cd plasma-python
python setup.py install
```

Where `my_env` should contain the Python packages as per `requirements-travis.txt` file.

#### Location of the data on Tigress

The JET and D3D datasets containing multi-modal time series of sensory measurements leading up to deleterious events called plasma disruptions are located on /tigress filesystem on Princeton U clusters.
Fo convenience, create following symbolic links:

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
This will preprocess the data and save it in `/tigress/<netid>/processed_shots` and `/tigress/<netid>/normalization`


#### Training and inference

Use Slurm scheduler to perform batch or interactive analysis on Tiger cluster.

##### Batch analysis

For batch analysis, make sure to allocate 1 process per GPU:

```bash
#!/bin/bash
#SBATCH -t 01:30:00
#SBATCH -N X
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH -c 4

module load anaconda3
source activate my_env
module load cudatoolkit/8.0 cudnn/cuda-8.0/6.0 openmpi/cuda-8.0/intel-17.0/2.1.0/64 intel/17.0/64/17.0.2.174
module load intel/17.0/64/17.0.4.196 intel-mkl/2017.3/4/64
srun python mpi_learn.py

```
where X is the number of nodes for distibuted training.

Submit the job with:
```bash
#cd examples
sbatch slurm.cmd
```

And monitor it's completion via:
```bash
squeue -u <netid>
```
Optionally, add an email notification option in the Slurm about the job completion.

##### Interactive analysis

Interactive option is preferred for debugging or running in the notebook, for all other case batch is preferred.
The workflow is to request an interactive session:

```bash
salloc -N [X] --ntasks-per-node=4 --ntasks-per-socket=2 --gres=gpu:4 -t 0-6:00
```
where the number of GPUs is X * 4.

Then launch the application from the command line:

```bash
mpirun -npernode 4 python examples/mpi_learn.py
```

### Understanding the data

All the configuration parameters are summarised in `examples/conf.yaml`. Highlighting the important ones to control the data.
Currently, FRNN is capable of working with JET and D3D data as well as cross-machine regime. The switch is done in the configuration file:

```yaml
paths:
    ... 
    data: 'jet_data'
```
use `d3d_data` for D3D signals, use `jet_to_d3d_data` ir `d3d_to_jet_data` for cross-machine regime.
    
By default, FRNN will select, preprocess and normalize all valid signals available. To chose only specific signals use:
```yaml
paths:
    ... 
    specific_signals: [q95,ip] 
```    
if left empty `[]` will use all valid signals defined on a machine. Only use if need a custom set.

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

The summaries are written real time to `/tigress/<netid>/Graph`. For MacOS, you can set up the `sshfs` mount of /tigress filesystem and view those summaries in your browser.

For Mac, you could follow the instructions here:
https://github.com/osxfuse/osxfuse/wiki/SSHFS

then do something like:
```
sshfs -o allow_other,defer_permissions netid@tigergpu.princeton.edu:/tigress/netid/ /mnt/<destination folder name on your laptop>/
```

Launch TensorBoard locally:
```
python -m tensorflow.tensorboard --logdir /mnt/<destination folder name on your laptop>/Graph
```
You should see something like:

![alt text](https://github.com/PPPLDeepLearning/plasma-python/blob/master/docs/tb.png)

#### Learning curves and ROC per epoch

Besides TensorBoard summaries you can produce the ROC curves for validation and test data as well as visualizations of shots:
```
cd examples/
python performance_analysis.py
```
this uses the resulting file produced as a result of training the neural network as an input, and produces several `.png` files with plots as an output.

In addition, you can check the scalar variable summaries for training loss, validation loss and validation ROC logged at `/tigress/netid/csv_logs` (each run will produce a new log file with a timestamp in name).

A sample code to analyze can be found in `examples/notebooks`. For instance:

```python
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file, save

data = pd.read_csv("/mnt/<destination folder name on your laptop>/csv_logs/<name of the log file>.csv")

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

To extract per mini-batch summaries, use the output produced by FRNN logged to the standard out (in case of the batch jobs, it will all be contained in the Slurm output file). Refer to the following notebook to perform the analysis of learning curve on a mini-batch level:
https://github.com/PPPLDeepLearning/plasma-python/blob/master/examples/notebooks/FRNN_scaling.ipynb
