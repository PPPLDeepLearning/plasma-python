import keras
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout, Lambda, Reshape, Flatten, Permute
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.utils.data_utils import get_file
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate
from keras.callbacks import Callback
from keras.optimizers import *
from keras.regularizers import l1,l2,l1_l2


import keras.backend as K

import dill
import re
import os,sys
import numpy as np
from copy import deepcopy

x = np.array(range(8))
num_signals = len(x)
x = np.atleast_2d(x)
x = np.reshape(x,(1,num_signals))
print(x.shape)
print(x)

indices_0d = np.array([0,1])
indices_1d = np.array([2,3,4,5,6,7])

num_1D = 2

pre_rnn_input = Input(shape=(num_signals,))
pre_rnn_1D = Lambda(lambda x: x[:,len(indices_0d):],output_shape=(len(indices_1d),))(pre_rnn_input)
pre_rnn_0D = Lambda(lambda x: x[:,:len(indices_0d)],output_shape=(len(indices_0d),))(pre_rnn_input)# slicer(x,indices_0d),lambda s: slicer_output_shape(s,indices_0d))(pre_rnn_input)
pre_rnn_1D = Reshape((num_1D,len(indices_1d)/num_1D)) (pre_rnn_1D)
pre_rnn_1D = Permute((2,1)) (pre_rnn_1D)
    
# for i in range(model_conf['num_conv_layers']):
#     pre_rnn_1D = Convolution1D(num_conv_filters,size_conv_filters,padding='valid',activation='relu') (pre_rnn_1D)
#     pre_rnn_1D = MaxPooling1D(pool_size) (pre_rnn_1D)
# pre_rnn_1D = Flatten() (pre_rnn_1D)
# pre_rnn = Concatenate() ([pre_rnn_0D,pre_rnn_1D])

model = Model(inputs = pre_rnn_input,outputs=pre_rnn_1D)
# x_input = Input(batch_shape = batch_input_shape)
# x_in = TimeDistributed(pre_rnn_model) (x_input)

# if return_sequences:
    #x_out = TimeDistributed(Dense(100,activation='tanh')) (x_in)
    # x_out = TimeDistributed(Dense(1,activation=output_activation)) (x_in)
# else:
    # x_out = Dense(1,activation=output_activation) (x_in)
model.compile(loss='mse',optimizer='sgd')

y = model.predict(x)
print(model.layers)
print(x)
print(y)
print(y.shape)
print(y[0,:,0])
#bug with tensorflow/Keras