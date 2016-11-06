# Model builder

Depends on Keras and model loader

Contains 2 classes: ModelBuilder and LossHistory (utility)

ModelBuilder takes conf and builds a model using Keras library, and provides methods to manipulate the model (save, load, etc)

## MPI builder

Serves a similar purpose, provides a set of MPI wrapper classes. Uses Keras SGD with Theano backend.

# Model runner

Depends on model Loader, performance utils

Contains a set of standalone functions, which givena shotlist perform training, make predictions, make evaluations and produce plots.


# Targets

Defines a class hierarchy of targets, specifying loss, activation functions and other params for the NNs


# Loader

Depends on from primitives.shots

Given conf and shotlist, provides tools to load shotlist, get batches, construct patches and manipulate them.

It is a way to deliver preprocessed data into model and prepare it for training.
