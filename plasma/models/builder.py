from __future__ import division
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import (
    Dense, Activation, Dropout, Lambda,
    Reshape, Flatten, Permute,  # RepeatVector
    )
from keras.layers import LSTM, SimpleRNN, BatchNormalization
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
# from keras.utils.data_utils import get_file
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate
from keras.callbacks import Callback
from keras.regularizers import l2  # l1, l1_l2

import keras.backend as K

import re
import os
import sys
import numpy as np
from copy import deepcopy
from plasma.utils.downloading import makedirs_process_safe, general_object_hash


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))


class ModelBuilder(object):
    def __init__(self, conf):
        self.conf = conf

    def get_unique_id(self):
        this_conf = deepcopy(self.conf)
        # ignore hash depednecy on number of epochs or T_min_warn (they are
        # both modifiable). Map local copy of all confs to the same values
        this_conf['training']['num_epochs'] = 0
        this_conf['data']['T_min_warn'] = 30
        unique_id = general_object_hash(this_conf)
        return unique_id

    def get_0D_1D_indices(self):
        # make sure all 1D indices are contiguous in the end!
        use_signals = self.conf['paths']['use_signals']
        indices_0d = []
        indices_1d = []
        num_0D = 0
        num_1D = 0
        curr_idx = 0
        # do we have any 1D indices?
        is_1D_region = use_signals[0].num_channels > 1
        for sig in use_signals:
            num_channels = sig.num_channels
            indices = range(curr_idx, curr_idx+num_channels)
            if num_channels > 1:
                indices_1d += indices
                num_1D += 1
                is_1D_region = True
            else:
                assert(not is_1D_region)
                # , "make sure all use_signals are ordered such that 1D signals
                # come last!"
                assert(num_channels == 1)
                indices_0d += indices
                num_0D += 1
                is_1D_region = False
            curr_idx += num_channels
        return np.array(indices_0d).astype(
            np.int32), np.array(indices_1d).astype(
            np.int32), num_0D, num_1D

    def build_model(self, predict, custom_batch_size=None):
        conf = self.conf
        model_conf = conf['model']
        rnn_size = model_conf['rnn_size']
        rnn_type = model_conf['rnn_type']
        regularization = model_conf['regularization']
        dense_regularization = model_conf['dense_regularization']
        use_batch_norm = False
        if 'use_batch_norm' in model_conf:
            use_batch_norm = model_conf['use_batch_norm']

        dropout_prob = model_conf['dropout_prob']
        length = model_conf['length']
        pred_length = model_conf['pred_length']
        # skip = model_conf['skip']
        stateful = model_conf['stateful']
        return_sequences = model_conf['return_sequences']
        # model_conf['output_activation']
        output_activation = conf['data']['target'].activation
        use_signals = conf['paths']['use_signals']
        num_signals = sum([sig.num_channels for sig in use_signals])
        num_conv_filters = model_conf['num_conv_filters']
        # num_conv_layers = model_conf['num_conv_layers']
        size_conv_filters = model_conf['size_conv_filters']
        pool_size = model_conf['pool_size']
        dense_size = model_conf['dense_size']

        batch_size = self.conf['training']['batch_size']
        if predict:
            batch_size = self.conf['model']['pred_batch_size']
            # so we can predict with one time point at a time!
            if return_sequences:
                length = pred_length
            else:
                length = 1

        if custom_batch_size is not None:
            batch_size = custom_batch_size

        if rnn_type == 'LSTM':
            rnn_model = LSTM
        elif rnn_type == 'SimpleRNN':
            rnn_model = SimpleRNN
        else:
            print('Unkown Model Type, exiting.')
            exit(1)

        batch_input_shape = (batch_size, length, num_signals)
        # batch_shape_non_temporal = (batch_size, num_signals)

        indices_0d, indices_1d, num_0D, num_1D = self.get_0D_1D_indices()

        def slicer(x, indices):
            return x[:, indices]

        def slicer_output_shape(input_shape, indices):
            shape_curr = list(input_shape)
            assert len(shape_curr) == 2  # only valid for 3D tensors
            shape_curr[-1] = len(indices)
            return tuple(shape_curr)

        pre_rnn_input = Input(shape=(num_signals,))

        if num_1D > 0:
            pre_rnn_1D = Lambda(lambda x: x[:, len(indices_0d):],
                                output_shape=(len(indices_1d),))(pre_rnn_input)
            pre_rnn_0D = Lambda(lambda x: x[:, :len(indices_0d)],
                                output_shape=(len(indices_0d),))(pre_rnn_input)
            # slicer(x,indices_0d),lambda s:
            # slicer_output_shape(s,indices_0d))(pre_rnn_input)
            pre_rnn_1D = Reshape((num_1D, len(indices_1d)//num_1D))(pre_rnn_1D)
            pre_rnn_1D = Permute((2, 1))(pre_rnn_1D)

            for i in range(model_conf['num_conv_layers']):
                div_fac = 2**i
                '''The first conv layer learns `num_conv_filters//div_fac`
                filters (aka kernels), each of size
                `(size_conv_filters, num1D)`. Its output will have shape
                (None, len(indices_1d)//num_1D - size_conv_filters + 1,
                num_conv_filters//div_fac), i.e., for
                each position in the input spatial series (direction along
                radius), the activation of each filter at that position.

                '''

                '''For i=1 first conv layer would get:
                (None, (len(indices_1d)//num_1D - size_conv_filters
                + 1)/pool_size-size_conv_filters + 1,num_conv_filters//div_fac)

                '''
                pre_rnn_1D = Convolution1D(
                    num_conv_filters//div_fac, size_conv_filters,
                    padding='valid')(pre_rnn_1D)
                if use_batch_norm:
                    pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
                pre_rnn_1D = Activation('relu')(pre_rnn_1D)

                '''The output of the second conv layer will have shape
                (None, len(indices_1d)//num_1D - size_conv_filters + 1,
                num_conv_filters//div_fac),
                i.e., for each position in the input spatial series
                (direction along radius), the activation of each filter
                at that position.

                For i=1, the second layer would output
                (None, (len(indices_1d)//num_1D - size_conv_filters + 1)/
                pool_size-size_conv_filters + 1,num_conv_filters//div_fac)
                '''
                pre_rnn_1D = Convolution1D(
                    num_conv_filters//div_fac, 1, padding='valid')(pre_rnn_1D)
                if use_batch_norm:
                    pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
                pre_rnn_1D = Activation('relu')(pre_rnn_1D)
                '''Outputs (None, (len(indices_1d)//num_1D - size_conv_filters
                + 1)/pool_size, num_conv_filters//div_fac)

                For i=1, the pooling layer would output:
                (None,((len(indices_1d)//num_1D- size_conv_filters
                + 1)/pool_size-size_conv_filters+1)/pool_size,
                num_conv_filters//div_fac)

                '''
                pre_rnn_1D = MaxPooling1D(pool_size)(pre_rnn_1D)
            pre_rnn_1D = Flatten()(pre_rnn_1D)
            pre_rnn_1D = Dense(
                dense_size,
                kernel_regularizer=l2(dense_regularization),
                bias_regularizer=l2(dense_regularization),
                activity_regularizer=l2(dense_regularization))(pre_rnn_1D)
            if use_batch_norm:
                pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
            pre_rnn_1D = Activation('relu')(pre_rnn_1D)
            pre_rnn_1D = Dense(
                dense_size//4,
                kernel_regularizer=l2(dense_regularization),
                bias_regularizer=l2(dense_regularization),
                activity_regularizer=l2(dense_regularization))(pre_rnn_1D)
            if use_batch_norm:
                pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
            pre_rnn_1D = Activation('relu')(pre_rnn_1D)
            pre_rnn = Concatenate()([pre_rnn_0D, pre_rnn_1D])
        else:
            pre_rnn = pre_rnn_input

        if model_conf['rnn_layers'] == 0 or (
                'extra_dense_input' in model_conf.keys()
                and model_conf['extra_dense_input']):
            pre_rnn = Dense(
                dense_size,
                activation='relu',
                kernel_regularizer=l2(dense_regularization),
                bias_regularizer=l2(dense_regularization),
                activity_regularizer=l2(dense_regularization))(pre_rnn)
            pre_rnn = Dense(
                dense_size//2,
                activation='relu',
                kernel_regularizer=l2(dense_regularization),
                bias_regularizer=l2(dense_regularization),
                activity_regularizer=l2(dense_regularization))(pre_rnn)
            pre_rnn = Dense(
                dense_size//4,
                activation='relu',
                kernel_regularizer=l2(dense_regularization),
                bias_regularizer=l2(dense_regularization),
                activity_regularizer=l2(dense_regularization))(pre_rnn)

        pre_rnn_model = Model(inputs=pre_rnn_input, outputs=pre_rnn)
        # pre_rnn_model.summary()
        x_input = Input(batch_shape=batch_input_shape)
        x_in = TimeDistributed(pre_rnn_model)(x_input)
        for _ in range(model_conf['rnn_layers']):
            x_in = rnn_model(
                rnn_size, return_sequences=return_sequences,
                # batch_input_shape=batch_input_shape,
                stateful=stateful, kernel_regularizer=l2(regularization),
                recurrent_regularizer=l2(regularization),
                bias_regularizer=l2(regularization), dropout=dropout_prob,
                recurrent_dropout=dropout_prob)(x_in)
            x_in = Dropout(dropout_prob)(x_in)
        if return_sequences:
            # x_out = TimeDistributed(Dense(100,activation='tanh')) (x_in)
            x_out = TimeDistributed(
                Dense(1, activation=output_activation))(x_in)
        else:
            x_out = Dense(1, activation=output_activation)(x_in)
        model = Model(inputs=x_input, outputs=x_out)
        # bug with tensorflow/Keras
        if (conf['model']['backend'] == 'tf'
                or conf['model']['backend'] == 'tensorflow'):
            first_time = "tensorflow" not in sys.modules
            import tensorflow as tf
            if first_time:
                K.get_session().run(tf.global_variables_initializer())

        model.reset_states()
        return model

    def build_train_test_models(self):
        return self.build_model(False), self.build_model(True)

    def save_model_weights(self, model, epoch):
        save_path = self.get_save_path(epoch)
        model.save_weights(save_path, overwrite=True)

    def delete_model_weights(self, model, epoch):
        save_path = self.get_save_path(epoch)
        assert(os.path.exists(save_path))
        os.remove(save_path)

    def get_save_path(self, epoch):
        unique_id = self.get_unique_id()
        return (self.conf['paths']['model_save_path']
                + 'model.{}._epoch_.{}.h5'.format(unique_id, epoch))

    def ensure_save_directory(self):
        prepath = self.conf['paths']['model_save_path']
        makedirs_process_safe(prepath)

    def load_model_weights(self, model, custom_path=None):
        if custom_path is None:
            epochs = self.get_all_saved_files()
            if len(epochs) == 0:
                print('no previous checkpoint found')
                return -1
            else:
                max_epoch = max(epochs)
                print('loading from epoch {}'.format(max_epoch))
                model.load_weights(self.get_save_path(max_epoch))
                return max_epoch
        else:
            epoch = self.extract_id_and_epoch_from_filename(
                os.path.basename(custom_path))[1]
            model.load_weights(custom_path)
            print("Loading from custom epoch {}".format(epoch))
            return epoch

    def get_latest_save_path(self):
        epochs = self.get_all_saved_files()
        if len(epochs) == 0:
            print('no previous checkpoint found')
            return ''
        else:
            max_epoch = max(epochs)
            print('loading from epoch {}'.format(max_epoch))
            return self.get_save_path(max_epoch)

    def extract_id_and_epoch_from_filename(self, filename):
        regex = re.compile(r'-?\d+')
        numbers = [int(x) for x in regex.findall(filename)]
        assert(len(numbers) == 3)  # id,epoch number and extension
        assert(numbers[2] == 5)  # .h5 extension
        return numbers[0], numbers[1]

    def get_all_saved_files(self):
        self.ensure_save_directory()
        unique_id = self.get_unique_id()
        path = self.conf['paths']['model_save_path']
        filenames = [name for name in os.listdir(path)
                     if os.path.isfile(os.path.join(path, name))]
        epochs = []
        for file in filenames:
            curr_id, epoch = self.extract_id_and_epoch_from_filename(file)
            if curr_id == unique_id:
                epochs.append(epoch)
        return epochs

    # FIXME this is essentially the ModelBuilder.build_model
        # in the long run we want to replace the space dictionary with the
        # regular conf file - I am sure there is a way to accomodate
    def hyper_build_model(self, space, predict, custom_batch_size=None):
        conf = self.conf
        model_conf = conf['model']
        rnn_size = model_conf['rnn_size']
        rnn_type = model_conf['rnn_type']
        regularization = model_conf['regularization']

        dropout_prob = model_conf['dropout_prob']
        length = model_conf['length']
        pred_length = model_conf['pred_length']
        # skip = model_conf['skip']
        stateful = model_conf['stateful']
        return_sequences = model_conf['return_sequences']
        # model_conf['output_activation']
        output_activation = conf['data']['target'].activation
        num_signals = conf['data']['num_signals']

        batch_size = self.conf['training']['batch_size']
        if predict:
            batch_size = self.conf['model']['pred_batch_size']
            # so we can predict with one time point at a time!
            if return_sequences:
                length = pred_length
            else:
                length = 1

        if custom_batch_size is not None:
            batch_size = custom_batch_size

        if rnn_type == 'LSTM':
            rnn_model = LSTM
        elif rnn_type == 'SimpleRNN':
            rnn_model = SimpleRNN
        else:
            print('Unkown Model Type, exiting.')
            exit(1)

        batch_input_shape = (batch_size, length, num_signals)
        model = Sequential()

        for _ in range(model_conf['rnn_layers']):
            model.add(
                rnn_model(
                    rnn_size,
                    return_sequences=return_sequences,
                    batch_input_shape=batch_input_shape,
                    stateful=stateful,
                    kernel_regularizer=l2(regularization),
                    recurrent_regularizer=l2(regularization),
                    bias_regularizer=l2(regularization),
                    dropout=dropout_prob,
                    recurrent_dropout=dropout_prob))
            model.add(Dropout(space['Dropout']))
        if return_sequences:
            model.add(TimeDistributed(Dense(1, activation=output_activation)))
        else:
            model.add(Dense(1, activation=output_activation))
        model.reset_states()

        return model
