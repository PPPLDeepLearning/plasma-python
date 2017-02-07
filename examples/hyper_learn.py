from __future__ import print_function
import sys
import matplotlib
matplotlib.use('Agg')

from plasma.conf import conf
from pprint import pprint
pprint(conf)

from plasma.primitives.shots import Shot, ShotList
import os

from hyperopt import Trials, tpe
from hyperopt import STATUS_OK
from hyperas import optim


def data_wrapper(conf):
	from plasma.models.loader import Loader
	from plasma.preprocessor.normalize import VarNormalizer as Normalizer
	import numpy as np
	import theano
	from keras.utils.generic_utils import Progbar
	from keras import backend as K

	from plasma.models.runner import make_predictions_and_evaluate_gpu
	from hyperas.distributions import choice, uniform, conditional

	from plasma.models.builder import ModelBuilder, LossHistory

	from keras.models import Sequential
	from keras.layers.core import Dense, Activation, Dropout
	from keras.layers.recurrent import LSTM, SimpleRNN
	from keras.layers.wrappers import TimeDistributed
	from keras.callbacks import Callback
	from keras.optimizers import SGD, Adam, RMSprop, Nadam
	from keras.regularizers import l1,l2,l1l2


    
def model_wrapper(conf):

    class HyperModelBuilder(ModelBuilder):
        def build_model(self, predict, custom_batch_size=None):
            conf = self.conf
            model_conf = conf['model']
            rnn_size = model_conf['rnn_size']
            rnn_type = model_conf['rnn_type']
            optimizer = model_conf['optimizer']
            lr = model_conf['lr']
            clipnorm = model_conf['clipnorm']
            regularization = model_conf['regularization']

            if optimizer == 'sgd':
                optimizer_class = SGD
            elif optimizer == 'adam':
                optimizer_class = Adam
            elif optimizer == 'rmsprop':
                optimizer_class = RMSprop
            elif optimizer == 'nadam':
                optimizer_class = Nadam
            else:
                optimizer = optimizer

            if lr is not None or clipnorm is not None:
                optimizer = optimizer_class(lr=lr, clipnorm=clipnorm)

            loss_fn = conf['data']['target'].loss
            dropout_prob = model_conf['dropout_prob']
            length = model_conf['length']
            pred_length = model_conf['pred_length']
            skip = model_conf['skip']
            stateful = model_conf['stateful']
            return_sequences = model_conf['return_sequences']
            output_activation = conf['data']['target'].activation
            num_signals = conf['data']['num_signals']

            batch_size = self.conf['training']['batch_size']
            if predict:
                batch_size = self.conf['model']['pred_batch_size']
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
                model.add(rnn_model(rnn_size, return_sequences=return_sequences, batch_input_shape=batch_input_shape,
                                    stateful=stateful, W_regularizer=l2(regularization),
                                    U_regularizer=l2(regularization),
                                    b_regularizer=l2(regularization), dropout_W=dropout_prob, dropout_U=dropout_prob))
                model.add(Dropout({{uniform(0, 1)}}))
            if return_sequences:
                model.add(TimeDistributed(Dense(1, activation=output_activation)))
            else:
                model.add(Dense(1, activation=output_activation))
            model.compile(loss=loss_fn, optimizer=optimizer)
            model.reset_states()
            return model


    nn = Normalizer(conf)
    nn.train()
    loader = Loader(conf,nn)
    shot_list_train,shot_list_validate,shot_list_test = loader.load_shotlists(conf)


    specific_builder = HyperModelBuilder(conf)
    train_model, test_model = specific_builder.build_model(False), specific_builder.build_model(True)    

    np.random.seed(1)
    validation_losses = []
    validation_roc = []
    training_losses = []
    shot_list_train,shot_list_validate = shot_list_train.split_direct(1.0-conf['training']['validation_frac'],do_shuffle=True)
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'

    num_epochs = conf['training']['num_epochs']
    num_at_once = conf['training']['num_shots_at_once']
    lr_decay = conf['model']['lr_decay']
    lr = conf['model']['lr']

    resulting_dict = {'loss':None,'status':STATUS_OK,'model':None}

    e = -1
    while e < num_epochs-1:
        e += 1
        pbar =  Progbar(len(shot_list_train))

        shot_list_train.shuffle()
        shot_sublists = shot_list_train.sublists(num_at_once)[:1]
        training_losses_tmp = []

        K.set_value(train_model.optimizer.lr, lr*lr_decay**(e))
        for (i,shot_sublist) in enumerate(shot_sublists):
            X_list,y_list = loader.load_as_X_y_list(shot_sublist)
            for j,(X,y) in enumerate(zip(X_list,y_list)):
                history = LossHistory()
                train_model.fit(X,y,
                    batch_size=Loader.get_batch_size(conf['training']['batch_size'],prediction_mode=False),
                    nb_epoch=1,shuffle=False,verbose=0,
                    validation_split=0.0,callbacks=[history])
                train_model.reset_states()
                train_loss = np.mean(history.losses)
                training_losses_tmp.append(train_loss)

                pbar.add(1.0*len(shot_sublist)/len(X_list), values=[("train loss", train_loss)])
                loader.verbose=False
        sys.stdout.flush()
        training_losses.append(np.mean(training_losses_tmp))
        specific_builder.save_model_weights(train_model,e)

        roc_area,loss = make_predictions_and_evaluate_gpu(conf,shot_list_validate,loader)
        validation_losses.append(loss)
        validation_roc.append(roc_area)
        resulting_dict['loss'] = loss
        resulting_dict['model'] = train_model  

    return resulting_dict

best_run, best_model = optim.minimize(model=model_wrapper,data=data_wrapper,algo=tpe.suggest,max_evals=2,trials=Trials())
print (best_run)
print (best_model)
