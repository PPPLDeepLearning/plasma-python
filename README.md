# FRNN [![Build Status](https://travis-ci.org/PPPLDeepLearning/plasma-python.svg?branch=master)](https://travis-ci.org/PPPLDeepLearning/plasma-python.svg?branch=master)

## Package description

Fusion Recurrent Neural Net (FRNN) is a Python package implementing deep learning models for disruption prediction in tokamak fusion plasmas.

It consists of 4 core modules:

- `models`: Python classes necessary to construct, train and optimize deep RNN models. Including a distributed data-parallel synchronous implementation of mini-batch gradient descent. FRNN makes use of MPI for communication and supports Tensorflow and Theano backends through Keras. FRNN allows running hyperparameter search optimizations

- `preprocessors`: signal preprocessing and normalization classes, including the methods necessary to prepare physical data for stateful LSTM training.

- `primitives`: contains abstractions specific to the domain, implemented as Python classes. For instance: Shot - a measurement of plasma current as a function of time. The Shot object contains attributes corresponding to unique identifier of a shot, disruption time in milliseconds, time profile of the shot converted to time-to- disruption values, validity of a shot (whether plasma current reaches a certain value during the shot), etc. Other primitives include `Machines` and `Signals` which carry the relevant information necessary for incorporating physics data into the overall pipeline. Signals know the Machine they live on, their mds+ paths, code for being downloaded, preprocessing approaches, their dimensionality, etc. Machines know which Signals are defined on them, which mds+ server houses the data, etc.

- `utilities`: a set of auxiliary functions for preprocessing, performance evaluation and learning curves analysis. 

In addition to the `utilities` FRNN supports TensorBoard scaler variable summaries, histogramms of layers, activations and gradients and graph visualizations.

This is a pure Python implementation for Python versions 2.7 and 3.6.

## Installation

The package comes with a standard setup script and a list of dependencies which include: mpi4py, TensorFlow, Theano,
Keras, h5py, Pathos. It also requires a standard set of CUDA drivers to run on GPU.

Then checkout the repo and use the setup script:

```bash
git clone https://github.com/PPPLDeepLearning/plasma-python
cd plasma-python
python setup.py install
```

with `sudo` if superuser permissions are needed or `--home=~` to install in a home directory. The latter option requires an appropriate `PYTHONPATH`.

Alternatively run (no need to checkout the repository in that case):
```bash
pip install -i https://testpypi.python.org/pypi plasma
```
optionally add `--user` to install in a home directory.


## Module index

## Tutorials

### Sample usage on Tigergpu

First, create an isolated Anaconda environment and load CUDA drivers:
```
#cd plasma-python
module load anaconda3
module load cudatoolkit/8.0 cudann/cuda-8.0/5.1 openmpi/intel-17.0/1.10.2/64 intel/17.0/64/17.0.2.174
conda create --name my_env --files requirements.txt
source activate my_env
```

Then install the plasma-python package:

```bash
source activate my_env
python setup.py install
```

Where `my_env` should contain the Python packages as per `requirements.txt` file.


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

module load anaconda
source activate my_env
module load cudatoolkit/8.0 cudann/cuda-8.0/5.1 openmpi/intel-17.0/1.10.2/64 intel/17.0/64/17.0.2.174

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

### Understanding the configuration files

All the configuration parameters are summarised in `examples/conf.yaml`. Highlighting the important ones:

```yaml
paths:
    ... 
    data: 'jet_data'

data:

    floatx: 'float64'


