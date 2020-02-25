from __future__ import division, print_function
import plasma.global_vars as g
# KGF: the first time Keras is ever imported via mpi_learn.py -> mpi_runner.py
import keras.backend as K
# KGF: see below synchronization--- output is launched here
from keras.models import Model  # , Sequential
# KGF: (was used only in hyper_build_model())
from keras.layers import Input
from keras.layers.core import (
    Dense, Activation, Dropout, Lambda,
    Reshape, Flatten, Permute,  # RepeatVector
    )
from keras.layers import LSTM, CuDNNLSTM, SimpleRNN, BatchNormalization
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
# from keras.utils.data_utils import get_file
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate
from keras.callbacks import Callback
from keras.regularizers import l2  # l1, l1_l2

import re
import os
import sys
import numpy as np
from copy import deepcopy
from plasma.utils.downloading import makedirs_process_safe
from plasma.utils.hashing import general_object_hash
from plasma.models.tcn import TCN
# TODO(KGF): consider using importlib.util.find_spec() instead (Py>3.4)
try:
    import keras2onnx
    import onnx
except ImportError:  # as e:
    _has_onnx = False
    # onnx = None
    # keras2onnx = None
else:
    _has_onnx = True

# Synchronize 2x stderr msg from TensorFlow initialization via Keras backend
# "Succesfully opened dynamic library... libcudart" "Using TensorFlow backend."
if g.comm is not None:
    g.flush_all_inorder()
# TODO(KGF): need to create wrapper .py file (or place in some __init__.py)
# that detects, for an arbitrary import, if tensorflow has been initialized
# either directly from "import tensorflow ..." and/or via backend of
# "from keras.layers ..."
# OR if this is the first time. See below "first_time" variable.


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
                assert not is_1D_region, (
                    "Check that use_signals are ordered with 1D signals last!")
                assert num_channels == 1
                indices_0d += indices
                num_0D += 1
                is_1D_region = False
            curr_idx += num_channels
        return (np.array(indices_0d).astype(np.int32),
                np.array(indices_1d).astype(np.int32), num_0D, num_1D)

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
        elif rnn_type == 'CuDNNLSTM':
            rnn_model = CuDNNLSTM
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
            if ('simple_conv' in model_conf.keys()
                    and model_conf['simple_conv'] is True):
                for i in range(model_conf['num_conv_layers']):
                    pre_rnn_1D = Convolution1D(
                        num_conv_filters, size_conv_filters,
                        padding='valid', activation='relu')(pre_rnn_1D)
                pre_rnn_1D = MaxPooling1D(pool_size)(pre_rnn_1D)
            else:
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
                    + 1)/pool_size-size_conv_filters + 1,
                    num_conv_filters//div_fac)

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
                        num_conv_filters//div_fac, 1, padding='valid')(
                            pre_rnn_1D)
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
        # TODO(KGF): uncomment following lines to get summary of pre-RNN model
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        # task_index = comm.Get_rank()
        # if not predict and task_index == 0:
        #     print('Printing out pre_rnn model...')
        #     fr = open('model_architecture.log', 'w')
        #     ori = sys.stdout
        #     sys.stdout = fr
        #     pre_rnn_model.summary()
        #     sys.stdout = ori
        #     fr.close()
        # pre_rnn_model.summary()
        x_input = Input(batch_shape=batch_input_shape)
        # TODO(KGF): Ge moved this inside a new conditional in Dec 2019. check
        # x_in = TimeDistributed(pre_rnn_model)(x_input)
        if (num_1D > 0 or (
                'extra_dense_input' in model_conf.keys()
                and model_conf['extra_dense_input'])):
            x_in = TimeDistributed(pre_rnn_model)(x_input)
        else:
            x_in = x_input

        # ==========
        # TCN MODEL
        # ==========
        if ('keras_tcn' in model_conf.keys()
                and model_conf['keras_tcn'] is True):
            print('Building TCN model....')
            tcn_layers = model_conf['tcn_layers']
            tcn_dropout = model_conf['tcn_dropout']
            nb_filters = model_conf['tcn_hidden']
            kernel_size = model_conf['kernel_size_temporal']
            nb_stacks = model_conf['tcn_nbstacks']
            use_skip_connections = model_conf['tcn_skip_connect']
            activation = model_conf['tcn_activation']
            use_batch_norm = model_conf['tcn_batch_norm']
            for _ in range(model_conf['tcn_pack_layers']):
                x_in = TCN(
                    use_batch_norm=use_batch_norm, activation=activation,
                    use_skip_connections=use_skip_connections,
                    nb_stacks=nb_stacks, kernel_size=kernel_size,
                    nb_filters=nb_filters, num_layers=tcn_layers,
                    dropout_rate=tcn_dropout)(x_in)
                x_in = Dropout(dropout_prob)(x_in)
        else:  # end TCN model
            # ==========
            # RNN MODEL
            # ==========
            # LSTM in ONNX: "The maximum opset needed by this model is only 9."
            model_kwargs = dict(return_sequences=return_sequences,
                                # batch_input_shape=batch_input_shape,
                                stateful=stateful,
                                kernel_regularizer=l2(regularization),
                                recurrent_regularizer=l2(regularization),
                                bias_regularizer=l2(regularization),
                                )
            if rnn_type != 'CuDNNLSTM':
                # Dropout is unsupported in CuDNN library
                model_kwargs['dropout'] = dropout_prob
                model_kwargs['recurrent_dropout'] = dropout_prob
            for _ in range(model_conf['rnn_layers']):
                x_in = rnn_model(rnn_size, **model_kwargs)(x_in)
                x_in = Dropout(dropout_prob)(x_in)
            if return_sequences:
                # x_out = TimeDistributed(Dense(100,activation='tanh')) (x_in)
                x_out = TimeDistributed(
                    Dense(1, activation=output_activation))(x_in)
        model = Model(inputs=x_input, outputs=x_out)
        # bug with tensorflow/Keras
        # TODO(KGF): what is this bug? this is the only direct "tensorflow"
        # import outside of mpi_runner.py and runner.py
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
        # try:
        if _has_onnx:
            save_path = self.get_save_path(epoch, ext='onnx')
            onnx_model = keras2onnx.convert_keras(model, model.name,
                                                  target_opset=10)
            onnx.save_model(onnx_model, save_path)
        # except Exception as e:
        #     print(e)
        return

    def delete_model_weights(self, model, epoch):
        save_path = self.get_save_path(epoch)
        assert os.path.exists(save_path)
        os.remove(save_path)

    def get_save_path(self, epoch, ext='h5'):
        unique_id = self.get_unique_id()
        dir_path = self.conf['paths']['model_save_path']
        # TODO(KGF): consider storing .onnx files in subdirectory away from .h5
        # if ext == 'onnx':
        #     os.path.join(dir_path, 'onnx/')
        return os.path.join(
            dir_path, 'model.{}._epoch_.{}.{}'.format(unique_id, epoch, ext))

    def ensure_save_directory(self):
        prepath = self.conf['paths']['model_save_path']
        makedirs_process_safe(prepath)

    def load_model_weights(self, model, custom_path=None):
        if custom_path is None:
            epochs = self.get_all_saved_files()
            if len(epochs) == 0:
                g.write_all('no previous checkpoint found\n')
                # TODO(KGF): port indexing change (from "return -1") to parts
                # of the code other than mpi_runner.py
                return 0
            else:
                max_epoch = max(epochs)
                g.write_all('loading from epoch {}\n'.format(max_epoch))
                model.load_weights(self.get_save_path(max_epoch))
                return max_epoch
        else:
            epoch = self.extract_id_and_epoch_from_filename(
                os.path.basename(custom_path))[1]
            model.load_weights(custom_path)
            g.write_all("Loading from custom epoch {}\n".format(epoch))
            return epoch

    # TODO(KGF): method was only called in non-MPI runner.py. Remove.
    # def get_latest_save_path(self):
    #     epochs = self.get_all_saved_files()
    #     if len(epochs) == 0:
    #         print('no previous checkpoint found')
    #         return ''
    #     else:
    #         max_epoch = max(epochs)
    #         print('loading from epoch {}'.format(max_epoch))
    #         return self.get_save_path(max_epoch)

    def extract_id_and_epoch_from_filename(self, filename):
        regex = re.compile(r'-?\d+')
        numbers = [int(x) for x in regex.findall(filename)]
        if filename[-3:] == '.h5':
            assert len(numbers) == 3  # id, epoch number, and .h5 extension
            assert numbers[2] == 5  # .h5 extension
        return numbers[0], numbers[1]

    def get_all_saved_files(self):
        self.ensure_save_directory()
        unique_id = self.get_unique_id()
        path = self.conf['paths']['model_save_path']
        # TODO(KGF): probably should only list .h5 file, not ONNX right now
        filenames = [name for name in os.listdir(path)
                     if os.path.isfile(os.path.join(path, name))]
        epochs = []
        for file in filenames:
            curr_id, epoch = self.extract_id_and_epoch_from_filename(file)
            if curr_id == unique_id:
                epochs.append(epoch)
        return epochs

    # TODO(felker): remove the following code or use as template for DeepHyper
    # plugin. Formerly was only used in single-GPU runner.py with hyperopt

    # TODO(alexeys): this is essentially the ModelBuilder.build_model
    # in the long run we want to replace the space dictionary with the
    # regular conf file - I am sure there is a way to accomodate
    # def hyper_build_model(self, space, predict, custom_batch_size=None):
    #     conf = self.conf
    #     model_conf = conf['model']
    #     rnn_size = model_conf['rnn_size']
    #     rnn_type = model_conf['rnn_type']
    #     regularization = model_conf['regularization']

    #     dropout_prob = model_conf['dropout_prob']
    #     length = model_conf['length']
    #     pred_length = model_conf['pred_length']
    #     # skip = model_conf['skip']
    #     stateful = model_conf['stateful']
    #     return_sequences = model_conf['return_sequences']
    #     # model_conf['output_activation']
    #     output_activation = conf['data']['target'].activation
    #     num_signals = conf['data']['num_signals']

    #     batch_size = self.conf['training']['batch_size']
    #     if predict:
    #         batch_size = self.conf['model']['pred_batch_size']
    #         # so we can predict with one time point at a time!
    #         if return_sequences:
    #             length = pred_length
    #         else:
    #             length = 1

    #     if custom_batch_size is not None:
    #         batch_size = custom_batch_size

    #     if rnn_type == 'LSTM':
    #         rnn_model = CuDNNLSTM
    #     elif rnn_type == 'SimpleRNN':
    #         rnn_model = SimpleRNN
    #     else:
    #         print('Unkown Model Type, exiting.')
    #         exit(1)

    #     batch_input_shape = (batch_size, length, num_signals)
    #     model = Sequential()

    #     for _ in range(model_conf['rnn_layers']):
    #         model.add(
    #             rnn_model(
    #                 rnn_size,
    #                 return_sequences=return_sequences,
    #                 batch_input_shape=batch_input_shape,
    #                 stateful=stateful,
    #                 kernel_regularizer=l2(regularization),
    #                 recurrent_regularizer=l2(regularization),
    #                 bias_regularizer=l2(regularization),
    #                 # dropout=dropout_prob,
    #                 # recurrent_dropout=dropout_prob
    #             ))
    #         model.add(Dropout(space['Dropout']))
    #     if return_sequences:
    #         model.add(TimeDistributed(Dense(1, activation=output_activation)
    # ))
    #     else:
    #         model.add(Dense(1, activation=output_activation))
    #     model.reset_states()

    #     return model
