# plasma-python
PPPL deep learning disruption prediction package

## Sample usage on Tiger

```bash
module load anaconda cudatoolkit/7.5 cudann openmpi/intel-16.0/1.8.8/64
source activate environment
python setup.py install
```

Where `environment` should contain the Python packages as per `requirements.txt` file.

### Preprocessing

```bash
python guarantee_preprocessed.py
```

### Training and inference

Use Slurm scheduler to perform batch or interactive analysis on Tiger cluster.

#### Batch analysis

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


#### Interactive analysis

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
