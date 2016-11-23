# plasma-python
PPPL deep learning disruption prediction package

## Sample usage on Tiger

```bash
module load anaconda cudatoolkit/7.5 cudann
source activate environment
python setup.py install
```

Where `environment` should contain the Python packages as per `requirements.txt` file.

### Preprocessing

```bash
python guarantee_preprocessed.py
```

### Training and inference

Performed on a cluster, scheduled via Slurm:

```bash
sbatch slurm.cmd
```
