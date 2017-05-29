#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --mem=64000
#SBATCH -o log.out
#SBATCH --gres=gpu:4


module load anaconda
module rm openmpi
srun python learn_processed.py
echo "done."
