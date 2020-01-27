# DeepFusionAESP (FRNN)

*Last updated 2020-01-23.*

## Description
The Fusion Recurrent Neural Net (FRNN) software is a Python package that
implements deep learning models for disruption prediction in tokamak fusion
plasmas. Using diagnostic time-series data collected over years of tokamak pulses, or
"shots", combined with labels of the disruption time (or "nondisruptive")
provided by machine experts, our software aims to predict disruptions at least
30 ms before their onset. Future extensions of the framework will attempt to
explain and classify the precursors to disruptions well in-advance of the event.

The main machine learning model used in the module (and behind the project's name) is
based on a recurrent neural network (RNN), specifically the long short-term memory (LSTM)
architecture. 1D (radial) profile data, if available and enabled by the user, is
incorporated through a series of `Conv1D` layers and concatenated to the 0D scalar signal
data at every timestep before being fed to the RNN layers. Future work will involve
incorporating 2D (radial x toroidal) data from the electron cyclotron emission imaging
(ECEi) advanced diagnostic in operation.

The implementation is built on Keras with
a pre-v2.x TensorFlow backend. A second implementation in PyTorch is also
available. The software can optionally employ data parallelism with distributed
synchronous training through `mpi4py`. We do not currently support Horovod, but
we expect to implement it soon.

Despite the name, the codebase also contains implementations of a variety of
"shallow" ML methods for comparison: random forests, SVMs, single hidden layer
perceptrons, and gradient-boosted trees. We have added, or are about to add,
deep learning methods that are not based on RNNs, e.g. TCN and Transformers.

Refer to Kates-Harbeck, et al. (2019), for additional information.

## Code and datasets
The public software repository is hosted on GitHub:
https://github.com/PPPLDeepLearning/plasma-python

The master host for the raw data used in the project is located on the
[`/tigress`](https://researchcomputing.princeton.edu/storage/tigress) GPFS file
system hosted by Princeton Research Computing. At the time of writing, it
consists of:
- 1.8 TB of `SHOT_ID.txt` files each containing the data from an individual
diagnostic signal data, during a single shot, for the JET, DIII-D, and NSTX
tokamaks.
```
/tigress/FRNN/signal_data/
/tigress/FRNN/signal_data_new_nov2019/
```
- A few MB of two-column (with 3 space delimiter) `.txt` files containing ID
numbers of shots and the labeled disruption time (or `-1.000000` if the shot is
nondisruptive).
```
/tigress/FRNN/shot_lists/
```

The data is accessible from the ALCF Theta and Cooley systems via the Lustre
file system. Periodically, the data from Princeton is copied via Globus to this
secondary store:
```
/lus/theta-fs0/projects/fusiondl_aesp/shot_lists/
/lus/theta-fs0/projects/fusiondl_aesp/signal_data/
```


### Processed signal data

Although the subset of raw data that is used to train a single FRNN
configuration is typically ~100s of GBs, the raw data must first be preprocessed
into individual `SHOT_ID.npz` files each containing a pickled `Shot` class
object which combines the resampled, cut, clipped, normalized, and transformed
diagnostic data that was originally distributed among multiple `SHOT_ID.txt`
files.

An example of such a set of preprocessed shot data files can be found here:
```
/lus/theta-fs0/projects/fusiondl_aesp/felker/processed_shots/signal_group_250640798211266795112500621861190558178
```

This pipeline greatly reduces the size of the input data used during training or
inference. For the 0D FRNN model applied to our original set of 3449 shots from DIII-D,
the preprocessed `SHOT_ID.npz` files occupy only a few GB, e.g.

## Building and running the software

A comprehensive tutorial for building and running the main FRNN model on the P100 GPUs of 
the [`Tiger` cluster](https://researchcomputing.princeton.edu/systems-and-services/available-systems/tiger) 
can be found in the main repository:
https://github.com/PPPLDeepLearning/plasma-python/blob/master/docs/PrincetonUTutorial.md

Similarly, we maintain documentation for a tutorial on ALCF Theta:
https://github.com/PPPLDeepLearning/plasma-python/blob/master/docs/ALCF.md

However, it is generally less up-to-date than the GPU documentation.
