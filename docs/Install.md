# Installing on Traverse
* Last updated 2022-04-18 *


## Building the package
### Login

First, login to the traverse cluster headnode via ssh:
```
ssh -XC <yourusername>@traverse.princeton.edu
```

The instructions below work with these modules (as of April 2022)
```
(frnn) [rkube@traverse site-packages]$ module list
Currently Loaded Modulefiles:
 1) anaconda3/2021.11   2) openmpi/cuda-11.0/gcc/4.0.4/64   3) cudatoolkit/11.0   4) cudnn/cuda-11.x/8.2.0  
```

Next, check out the source code from github and enter the directory of the local copy:
```
git clone git://github.com/PPPLDeepLearning/plasma-python
cd plasma-python
```


After that, create a new conda environment and install the relevant packages, starting with pytorch. 
```
conda create --name conda create --name frnn --channel https://opence.mit.edu/#/ pytorch
```

Next, activate the envrionment can add other required packages
```
conda activate frnn
conda install pandas scipy flake8 h5py pyparsing pyyaml cython matplotlib scikit-learn joblib xgboost
```

Now we can install `plasma-python`:
```
python setup.py install
```

Once this command is done, the module is availabe in python from within the environment
```
(frnn) [rkube@traverse ~]$ ipython
importPython 3.8.13 (default, Mar 28 2022, 11:00:56) 
Type 'copyright', 'credits' or 'license' for more information
IPython 8.2.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import plasma

In [2]: print(plasma.__file__)
/home/user/.conda/envs/frnn/lib/python3.8/site-packages/plasma-1.0.0-py3.8.egg/plasma/__init__.py
```


