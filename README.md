# FRNN [![Build Status](https://travis-ci.org/PPPLDeepLearning/plasma-python.svg?branch=master)](https://travis-ci.org/PPPLDeepLearning/plasma-python.svg?branch=master)

## Package description

Fusion Recurrent Neural Net (FRNN) is a Python package implementing deep learning models for disruption prediction in tokamak fusion plasmas.

It consists of 4 core modules:

- models: Python classes necessary to construct, train and optimize deep RNN models. Including a distributed data-parallel synchronous implementation of mini-batch gradient descent. FRNN makes use of MPI for communication and supports Tensorflow and Theano backends through Keras

- preprocessors: signal preprocessing and normalization classes, including the methods necessary to prepare physical data for stateful LSTM training.

- primitives: contains abstractions specific to the domain implemented as Python classes. For instance: Shot - a measurement of plasma current as a function of time. The Shot object contains attributes corresponding to unique identifier of a shot, disruption time in milliseconds, time profile of the shot converted to time-to- disruption values, validity of a shot (whether plasma current reaches a certain value during the shot), etc

- utilities: a set of auxiliary functions for preprocessing, performance evaluation and learning curves analysis

This is a pure Python implementation for Python versions 2.7 and 3.6.

## Installation

The package comes with a standard setup script and a list of dependencies which include: mpi4py, Theano,
Keras, h5py, Pathos. It also requires a standard set of CUDA drivers to run on GPU.

Run:
```bash
pip install -i https://testpypi.python.org/pypi plasma
```
optionally add `--user` to install in a home directory.

Alternatively, use the setup script:

```bash
python setup.py install
```

with `sudo` if superuser permissions are needed or `--home=~` to install in a home directory. The latter option requires an appropriate `PYTHONPATH`.

## Module index

## Tutorials

### Sample usage on Tiger

```bash
module load anaconda cudatoolkit/7.5 cudann openmpi/intel-16.0/1.8.8/64
source activate environment
python setup.py install
```

Where `environment` should contain the Python packages as per `requirements.txt` file.

#### Preprocessing

```bash
python guarantee_preprocessed.py
```

#### Training and inference

Use Slurm scheduler to perform batch or interactive analysis on Tiger cluster.

##### Batch analysis

For batch analysis, make sure to allocate 1 process per GPU:

```bash
#SBATCH -N X
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
```
where X is the number of nodes for distibuted data parallel training.


```bash
sbatch slurm.cmd
```

##### Interactive analysis

The workflow is to request an interactive session:

```bash
salloc -N [X] --ntasks-per-node=16 --ntasks-per-socket=8 --gres=gpu:4 -t 0-6:00
```
where the number of GPUs is X * 4.


Then launch the application from the command line:

```bash
cd plasma-python
mpirun -npernode 4 python examples/mpi_learn.py
```

Note: there is Theano compilation going on in the 1st epoch which will distort timing. It is recommended to perform testing setting `num_epochs >= 2` in `conf.py`.


## Status
