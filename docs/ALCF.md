# ALCF Theta `plasma-python` FRNN Notes

**Original author: Rick Zamora (rzamora@anl.gov)**

This document is intended to act as a tutorial for running the [plasma-python](https://github.com/PPPLDeepLearning/plasma-python) implementation of the fusion recurrent neural network (FRNN) on the ALCF Theta supercomputer (Cray XC40; Intel KNL processors).  The steps followed in these notes are based on the Princeton [Tiger-GPU tutorial](https://github.com/PPPLDeepLearning/plasma-python/blob/master/docs/PrincetonUTutorial.md#location-of-the-data-on-tigress), hosted within the main GitHub repository for the project.

## Environment Setup

Choose a root project directory for FRNN-related installations on Theta:

```
export FRNN_ROOT=<desired-root-directory>
cd $FRNN_ROOT
```

*Personal note:* I am using `FRNN_ROOT=/home/zamora/ESP`

Create a simple directory structure allowing experimental *builds* of the `plasma-python` python code/library:

```
mkdir build
mkdir build/miniconda-3.6-4.5.4
cd build/miniconda-3.6-4.5.4
```

### Custom Miniconda Environment Setup

Copy miniconda installation script to working directory (and install):

```
cp /lus/theta-fs0/projects/fusiondl_aesp/FRNN/rzamora/scripts/install_miniconda-3.6-4.5.4.sh .
./install_miniconda-3.6-4.5.4.sh
```

The `install_miniconda-3.6-4.5.4.sh` script will install `miniconda-4.5.4` (using `Python-3.6`), as well as `Tensorflow-1.12.0` and `Keras 2.2.4`.


Update your environment variables to use miniconda:

```
export PATH=${FRNN_ROOT}/build/miniconda-3.6-4.5.4/miniconda3/4.5.4/bin:$PATH
export PYTHONPATH=${FRNN_ROOT}/build/miniconda-3.6-4.5.4/miniconda3/4.5.4/lib/python3.6/site-packages/:$PYTHONPATH
```

Note that the previous lines (as well as the definition of `FRNN_ROOT`) can be appended to your `$HOME/.bashrc` file if you want to use this environment on Theta by default.


## Installing `plasma-python`

Here, we assume the installation is within the custom miniconda environment installed in the previous steps. We also assume the following commands have already been executed:

```
export FRNN_ROOT=<desired-root-directory>
export PATH=${FRNN_ROOT}/build/miniconda-3.6-4.5.4/miniconda3/4.5.4/bin:$PATH
export PYTHONPATH=${FRNN_ROOT}/build/miniconda-3.6-4.5.4/miniconda3/4.5.4/lib/python3.6/site-packages/:$PYTHONPATH
```

*Personal note:* I am using `export FRNN_ROOT=/lus/theta-fs0/projects/fusiondl_aesp/zamora/FRNN_project`

If the environment is set up correctly, installation of `plasma-python` is straightforward:

```
cd ${FRNN_ROOT}/build/miniconda-3.6-4.5.4
git clone https://github.com/PPPLDeepLearning/plasma-python.git
cd plasma-python
python setup.py build
python setup.py install
```

## Data Access

Sample data and metadata is available in `/lus/theta-fs0/projects/FRNN/tigress/alexeys/signal_data` and `/lus/theta-fs0/projects/FRNN/tigress/alexeys/shot_lists`, respectively.  It is recommended that users create their own symbolic links to these directories. I recommend that you do this within a directory called `/lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/`. For example:

```
ln -s /lus/theta-fs0/projects/fusiondl_aesp/FRNN/tigress/alexeys/shot_lists  /lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/shot_lists
ln -s /lus/theta-fs0/projects/fusiondl_aesp/FRNN/tigress/alexeys/signal_data  /lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/signal_data
```

For the examples included in `plasma-python`, there is a configuration file that specifies the root directory of the raw data. Change the `fs_path: '/tigress'` line in `examples/conf.yaml` to reflect the following:

```
fs_path: '/lus/theta-fs0/projects/fusiondl_aesp'
```

Its also a good idea to change `num_gpus: 4` to `num_gpus: 1`. I am also using the `jet_data_0D` dataset:

```
paths:
    data: jet_data_0D
```


### Data Preprocessing

#### The SLOW Way (On Theta)

Theta is KNL-based, and is **not** the best resource for processing many text files in python. However, the preprocessing step *can* be used by using the following steps (although it may need to be repeated many times to get through the whole dataset in a 60-minute debug queues):

```
cd ${FRNN_ROOT}/build/miniconda-3.6-4.5.4/plasma-python/examples
cp /lus/theta-fs0/projects/fusiondl_aesp/FRNN/rzamora/scripts/submit_guarantee_preprocessed.sh .
```

Modify the paths defined in `submit_guarantee_preprocessed.sh` to match your environment.

Note that the preprocessing module will use Pathos multiprocessing (not MPI/mpi4py).  Therefore, the script will see every compute core (all 256 per node) as an available resource.  Since the LUSTRE file system is unlikely to perform well with 256 processes (on the same node) opening/closing/creating files at once, it might improve performance if you make a slight change to line 85 in the `vi ~/plasma-python/plasma/preprocessor/preprocess.py` file:

```
line 85: use_cores = min( <desired-maximum-process-count>, max(1,mp.cpu_count()-2) )
```

After optionally re-building and installing plasm-python with this change, submit the preprocessing job:

```
qsub submit_guarantee_preprocessed.sh
```

#### The FAST Way (On Cooley)

You will fine it much less painful to preprocess the data on Cooley, because the Haswell processors are much better suited for this... Log onto the ALCF Cooley Machine:

```
ssh <alcf-username>@cooley.alcf.anl.gov
```

Copy my `cooley_preprocess` example directory to whatever directory you choose to work in:

```
cp -r /lus/theta-fs0/projects/fusiondl_aesp/FRNN/rzamora/scripts/cooley_preprocess .
cd cooley_preprocess
```

This directory has a Singularity image with everything you need to run your code on Cooley. Assuming you have created symbolic links to the `shot_lists` and `signal_data` directories in `/lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/`, you can just submit the included `COBALT` script (to specify the data you want to process, just modify the included `conf.yaml` file):

```
qsub submit.sh
```

For me, this finishes in less than 10 minutes, and creates 5523 `.npz` files in the `/lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/processed_shots/` directory.  The output file of the COBALT submission ends with the following message:

```
5522/5523Finished Preprocessing 5523 files in 406.94421911239624 seconds
Omitted 5523 shots of 5523 total.
0/0 disruptive shots
WARNING: All shots were omitted, please ensure raw data is complete and available at /lus/theta-fs0/projects/fusiondl_aesp/zamora/signal_data/.
4327 1196
```


# Notes on Revisiting Pre-Processes

## Preprocessing Information

To understand what might be going wrong with the preprocessing step, let's investigate what the code is actually doing.

**Step 1** Call `guarentee_preprocessed( conf )`, which is defined in `plasma/preprocessor/preprocess.py`. This function first initializes a `Preprocessor()` object (whose class definition is in the same file), and then checks if the preprocessing was already done (by looking for a file). The preprocessor object is called `pp`.

**Step 2** Assuming preprocessing is needed, we call `pp.clean_shot_lists()`, which loops through each file in the `shot_lists` directory and calls `self.clean_shot_list()` (not plural) for each text-file item. I do not believe this function is doing any thing when I run it, because all the shot list files have been "cleaned." The cleaning of a shot-list file just means the data is corrected to have two columns, and the file is renamed (to have "clear" in the name).

**Step 3** We call `pp.preprocess_all()`, which parses some of the config file, and ultimately calls `self.preprocess_from_files(shot_files_all,use_shots)` (where I believe `shot_files_all` is the output directory, and `use_shots` is the number of shots to use).

**Step 4** The `preprocess_from_files()` function is used to do the actual preprocessing. It does this by creating a multiprocessing pool, and mapping the processes to the `self.preprocess_single_file` function (note that the code for `ShotList` class is in `plasma/primitives/shots.py`, and the preprocessing code is still in `plasma/preprocessor/preprocess.py`).

**Important:** It looks like the code uses the path definitions in `data/shot_lists/signals.py` to define the location/path of signal data. I believe that some of the signal data is missing, which is causing every "shot" to be labeled as incomplete (and consequently thrown out).

### Possible Issues

From the preprocessing output, it is clear that the *Signal Radiated Power Core* data was not downloaded correctly. According to the `data/shot_lists/signals.py` file, the data *should* be in `/lus/theta-fs0/projects/fusiondl_aesp/<alcf-user-name>/signal_data/jet/ppf/bolo/kb5h/channel14`. However, the only subdirectory of `~/jet/ppf/` is `~/jet/ppf/efit`

Another possible issue is that the `data/shot_lists/signals.py` file specifies the **name** of the directory containing the *Radiated Power* data incorrectly (*I THINK*). Instead of the following line:

`pradtot = Signal("Radiated Power",['jpf/db/b5r-ptot>out'],[jet])`

We might need this:

`pradtot = Signal("Radiated Power",['jpf/db/b5r-ptot\>out'],[jet])`

The issue has to do with the `>` character in the directory name (without the proper `\` escape character, python may be looking in the wrong path). **NOTE: I need to confirm that there is actually an issue with the way the code is actually using the string.**


## Singularity/Docker Notes

Recall that the data preprocessing step was PAINFULLY slow on Theta, and so I decided to use Cooley. To simplify the process of using Cooley, I created a Docker image with the necessary environment. 

*Personal Note:* I performed this work on my local machine (Mac) in `/Users/rzamora/container-recipes`.

In order to use a Docker image within a Singularity container (required on ALCF machines), it is useful to build the image on your local machine and push it to "Docker Hub":


**Step 1:** Install Docker if you don't have it. [Docker-Mac](https://www.docker.com/docker-mac) works well for Mac.

**Step 2:** Build a Docker image using the recipe discussed below.

```
export IMAGENAME="test_image"
export RECIPENAME="Docker.centos7-cuda-tf1.12.0"
docker build -t $IMAGENAME -f $RECIPENAME .
```

You can check that the image is functional by starting an interactive shell session, and checking that the necessary python modules are available. For example (using `-it` for an interactive session):

```
docker run --rm -it -v $PWD:/tmp -w /tmp $IMAGENAME:latest bash
# python -c "import keras; import plasma; print(plasma.__file__)"
```

Note that the `plasma-python` source code will be located in `/root/plasma-python/` for the recipe described below.

**Step 3:** Push the image to [Docker Hub](https://hub.docker.com/).

Using your docker-hub username:

```
docker login --username=<username>
```

Then, "tag" the image using the `IMAGE ID` value displayed with `docker image ls`:

```
docker tag <IMAGE-ID> <username>/<image-name>:<label>
```

Here, `<label>` is something like "latest".  To finally push the image to [Docker Hub](https://hub.docker.com/):

```
docker push <username>/<image-name>
```

### Docker Recipe

The actual content of the docker recipe is mostly borrowed from an example on [GitHub](https://github.com/scieule/golden-heart/blob/master/Dockerfile):

```
FROM nvidia/cuda:9.1-cudnn7-devel-centos7

# Setup environment:
SHELL ["/bin/bash", "-c"]
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV CUDA_DEVICE_ORDER PCI_BUS_ID
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/cuda/extras/CUPTI/lib64

RUN yum update -y

RUN yum groupinstall -y "Development tools"

RUN yum install -y  wget \
                    unzip \
                    screen tmux \
                    ruby \
                    vim \
                    bc \
                    man \
                    ncurses-devel \
                    zlib-devel \
                    curl-devel \
                    openssl-devel \
                    which

RUN yum install -y qt5*devel gtk2-devel

RUN yum install -y  blas-devel \
                    lapack-devel \
                    atlas-devel \
                    gcc-gfortran \
                    tbb-devel \
                    eigen3-devel \
                    jasper-devel \
                    libpng-devel \
                    libtiff-devel \
                    openexr-devel \
                    libwebp-devel \
                    libv4l-devel \
                    libdc1394-devel \
                    libv4l-devel \
                    gstreamer-plugins-base-devel

# C/C++ CMake Python
RUN yum install -y  centos-release-scl && \
    yum install -y  devtoolset-7-gcc* \
                    devtoolset-7-valgrind \
                    devtoolset-7-gdb \
                    devtoolset-7-elfutils \
                    clang \
                    llvm-toolset-7 \
                    llvm-toolset-7-cmake \
                    rh-python36-python-devel \
                    rh-python36-python-pip \
                    rh-git29-git \
                    devtoolset-7-make

RUN echo "source scl_source enable devtoolset-7" >> /etc/bashrc
RUN echo "source scl_source enable llvm-toolset-7" >> /etc/bashrc
RUN echo "source scl_source enable rh-python36" >> /etc/bashrc
RUN echo "source scl_source enable rh-git29" >> /etc/bashrc

# Python libs & jupyter

RUN source /etc/bashrc; pip3 install --upgrade pip
RUN source /etc/bashrc; pip3 install numpy scipy matplotlib pandas \
                                    tensorflow-gpu keras h5py tables \
                                    scikit-image scikit-learn Pillow opencv-python \
                                    jsonschema jinja2 tornado pyzmq ipython jupyter notebook

# Install MPICH
RUN  cd /root && wget -q http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz \
  && tar xf mpich-3.2.1.tar.gz \
  && rm mpich-3.2.1.tar.gz \
  && cd mpich-3.2.1 \
  && source /etc/bashrc; ./configure --prefix=/usr/local/mpich/install --disable-wrapper-rpath \
  && make -j 4 install \
  && cd .. \
  && rm -rf mpich-3.2.1

ENV PATH ${PATH}:/usr/local/mpich/install/bin
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/mpich/install/lib
RUN env | sort

# Install plasma-python (https://github.com/PPPLDeepLearning/plasma-python)
# For 'pip'-based install: pip --no-cache-dir --disable-pip-version-check install -i https://testpypi.python.org/pypi plasma
RUN cd /root && git clone https://github.com/PPPLDeepLearning/plasma-python \
  && cd plasma-python \
  && source /etc/bashrc; python setup.py install \
  && cd ..

# nccl2
RUN cd /root && git clone https://github.com/NVIDIA/nccl.git \
  && cd nccl \
  && make -j src.build \
  && make pkg.redhat.build \
  && rpm -i build/pkg/rpm/x86_64/libnccl*

# pip-install mpi4py
RUN source /etc/bashrc; pip3 install mpi4py

RUN yum install -y libffi libffi-devel

RUN source /etc/bashrc; pip3 install tensorflow

# Workaround to build horovod without needing cuda libraries available:
# temporary add stub drivers to ld.so.cache
RUN ldconfig /usr/local/cuda/lib64/stubs \
  && source /etc/bashrc; HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_NCCL_HOME=/nccl/build/ pip3 --no-cache-dir install horovod \
  && ldconfig

ENV NCCL_P2P_DISABLE 1
```

### Converting Docker to Singularity

Needed to build a singularity image for Cooley... Used vagrant:

```
cd ~/vm-singularity/
vagrant up
vagrant ssh
sudo singularity build centos7-cuda-tf1.12.0-plasma.simg docker://rjzamora/centos7-cuda-tf1.12.0.dimg:latest
```





# First time setup on Theta (Fall 2017)

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
