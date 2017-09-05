#First time setup on Theta, Argonne

```bash
mkdir PPPL
cd PPPL/
git clone https://github.com/PPPLDeepLearning/plasma-python

wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
sh Anaconda3-4.4.0-Linux-x86_64.sh 
PPPL/plasma-python/

conda create --name PPPL_dev --file=requirements-travis.txt 
#~/.bashrc
export PATH="/home/alexeys/anaconda3/bin:$PATH"
conda create --name PPPL_dev --file=requirements-travis.txt 
source activate PPPL_dev

python setup.py install
module load PrgEnv-intel/6.0.4
#which mpicc
env MPICC=/opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/bin/mpicc pip install --user mpi4py
