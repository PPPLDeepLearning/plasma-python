from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from itertools import imap

from hyperopt import hp, STATUS_OK
from hyperas.distributions import conditional

import time
import sys
import os
from functools import partial
import pathos.multiprocessing as mp

from plasma.conf import conf
from plasma.models.loader import Loader
from plasma.utils.performance import PerformanceAnalyzer
from plasma.utils.evaluation import *

def train(conf,shot_list_train,loader):

    np.random.seed(1)

    validation_losses = []
    validation_roc = []
    training_losses = []
    if conf['training']['validation_frac'] > 0.0:
        shot_list_train,shot_list_validate = shot_list_train.split_direct(1.0-conf['training']['validation_frac'],do_shuffle=True)
        print('validate: {} shots, {} disruptive'.format(len(shot_list_validate),shot_list_validate.num_disruptive()))
    print('training: {} shots, {} disruptive'.format(len(shot_list_train),shot_list_train.num_disruptive()))
    ##Need to import later because accessing the GPU from several processes via multiprocessing
    ## gives weird errors.
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'#,mode=NanGuardMode'
    import theano
    from keras.utils.generic_utils import Progbar 
    from keras import backend as K
    from plasma.models import builder #from model_builder import ModelBuilder, LossHistory

    print('Build model...',end='')
    specific_builder = builder.ModelBuilder(conf)
    train_model = specific_builder.build_model(False) 
    print('...done')

    #load the latest epoch we did. Returns -1 if none exist yet
    e = specific_builder.load_model_weights(train_model)

    num_epochs = conf['training']['num_epochs']
    num_at_once = conf['training']['num_shots_at_once']
    lr_decay = conf['model']['lr_decay']
    lr = conf['model']['lr']
    print('{} epochs left to go'.format(num_epochs - 1 - e))
    while e < num_epochs-1:
        e += 1
        print('\nEpoch {}/{}'.format(e+1,num_epochs))
        pbar =  Progbar(len(shot_list_train))

        #shuffle during every iteration
        shot_list_train.shuffle() 
        shot_sublists = shot_list_train.sublists(num_at_once)
        training_losses_tmp = []

        #decay learning rate each epoch:
        K.set_value(train_model.optimizer.lr, lr*lr_decay**(e))
        print('Learning rate: {}'.format(train_model.optimizer.lr.get_value()))
        for (i,shot_sublist) in enumerate(shot_sublists):
            X_list,y_list = loader.load_as_X_y_list(shot_sublist)
            for j,(X,y) in enumerate(zip(X_list,y_list)):
                history = builder.LossHistory()
                #load data and fit on data
                train_model.fit(X,y,
                    batch_size=Loader.get_batch_size(conf['training']['batch_size'],prediction_mode=False),
                    epochs=1,shuffle=False,verbose=0,
                    validation_split=0.0,callbacks=[history])
                train_model.reset_states()
                train_loss = np.mean(history.losses)
                training_losses_tmp.append(train_loss)

                pbar.add(1.0*len(shot_sublist)/len(X_list), values=[("train loss", train_loss)])
                loader.verbose=False#True during the first iteration
        sys.stdout.flush()
        training_losses.append(np.mean(training_losses_tmp))
        specific_builder.save_model_weights(train_model,e)

        if conf['training']['validation_frac'] > 0.0:
            roc_area,loss = make_predictions_and_evaluate_gpu(conf,shot_list_validate,loader)
            validation_losses.append(loss)
            validation_roc.append(roc_area)

        print('=========Summary========')
        print('Training Loss: {:.3e}'.format(training_losses[-1]))
        if conf['training']['validation_frac'] > 0.0:
            print('Validation Loss: {:.3e}'.format(validation_losses[-1]))
            print('Validation ROC: {:.4f}'.format(validation_roc[-1]))


    # plot_losses(conf,[training_losses],specific_builder,name='training')
    if conf['training']['validation_frac'] > 0.0:
        plot_losses(conf,[training_losses,validation_losses,validation_roc],specific_builder,name='training_validation_roc')
    print('...done')

class HyperRunner(object):
    def __init__(self,conf,loader,shot_list):
        self.loader = loader
        self.shot_list = shot_list
        self.conf = conf

    #FIXME setup for hyperas search
    def keras_fmin_fnct(self,space):
        from plasma.models import builder

        specific_builder = builder.ModelBuilder(self.conf)

        train_model, test_model = specific_builder.hyper_build_model(space,False), specific_builder.hyper_build_model(space,True)

        np.random.seed(1)
        validation_losses = []
        validation_roc = []
        training_losses = []
        shot_list_train,shot_list_validate = self.shot_list.split_direct(1.0-conf['training']['validation_frac'],do_shuffle=True)
        os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
	import theano
	from keras.utils.generic_utils import Progbar
	from keras import backend as K


        num_epochs = self.conf['training']['num_epochs']
        num_at_once = self.conf['training']['num_shots_at_once']
        lr_decay = self.conf['model']['lr_decay']
        lr = self.conf['model']['lr']

        resulting_dict = {'loss':None,'status':STATUS_OK,'model':None}

        e = -1
        #print("Current num_epochs {}".format(e))
        while e < num_epochs-1:
            e += 1
            pbar =  Progbar(len(shot_list_train))

            shot_list_train.shuffle()
            shot_sublists = shot_list_train.sublists(num_at_once)[:1]
            training_losses_tmp = []

            K.set_value(train_model.optimizer.lr, lr*lr_decay**(e))
            for (i,shot_sublist) in enumerate(shot_sublists):
                X_list,y_list = self.loader.load_as_X_y_list(shot_sublist)
                for j,(X,y) in enumerate(zip(X_list,y_list)):
                    history = builder.LossHistory()
                    train_model.fit(X,y,
                        batch_size=Loader.get_batch_size(self.conf['training']['batch_size'],prediction_mode=False),
                        epochs=1,shuffle=False,verbose=0,
                        validation_split=0.0,callbacks=[history])
                    train_model.reset_states()
                    train_loss = np.mean(history.losses)
                    training_losses_tmp.append(train_loss)

                    pbar.add(1.0*len(shot_sublist)/len(X_list), values=[("train loss", train_loss)])
                    self.loader.verbose=False
            sys.stdout.flush()
            training_losses.append(np.mean(training_losses_tmp))
            specific_builder.save_model_weights(train_model,e)

            roc_area,loss = make_predictions_and_evaluate_gpu(self.conf,shot_list_validate,self.loader)
            print("Epoch: {}, loss: {}, validation_losses_size: {}".format(e,loss,len(validation_losses)))
            validation_losses.append(loss)
            validation_roc.append(roc_area)
            resulting_dict['loss'] = loss
            resulting_dict['model'] = train_model
            #print("Results {}, before {}".format(resulting_dict,id(resulting_dict)))

        #print("Results {}, after {}".format(resulting_dict,id(resulting_dict)))
        return resulting_dict

    def get_space(self):
        return {
            'Dropout': hp.uniform('Dropout', 0, 1),
        }

    def frnn_minimize(self, algo, max_evals, trials, rseed=1337):
	from hyperopt import fmin

        best_run = fmin(self.keras_fmin_fnct,
                    space=self.get_space(),
                    algo=algo,
                    max_evals=max_evals,
                    trials=trials,
                    rstate=np.random.RandomState(rseed))

        best_model = None
        for trial in trials:
            vals = trial.get('misc').get('vals')
            for key in vals.keys():
                vals[key] = vals[key][0]
            if trial.get('misc').get('vals') == best_run and 'model' in trial.get('result').keys():
                best_model = trial.get('result').get('model')

        return best_run, best_model

def plot_losses(conf,losses_list,specific_builder,name=''):
    unique_id = specific_builder.get_unique_id()
    savedir = 'losses'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    save_path = os.path.join(savedir,'{}_loss_{}.png'.format(name,unique_id))
    plt.figure()
    for losses in losses_list:
        plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(save_path)





def make_predictions(conf,shot_list,loader):

    os.environ['THEANO_FLAGS'] = 'device=cpu' #=cpu
    import theano
    from keras.utils.generic_utils import Progbar 
    from plasma.models.builder import ModelBuilder
    specific_builder = ModelBuilder(conf) 
    


    y_prime = []
    y_gold = []
    disruptive = []

    model = specific_builder.build_model(True)
    specific_builder.load_model_weights(model)
    model_save_path = specific_builder.get_latest_save_path()

    start_time = time.time()
    use_cores = max(1,mp.cpu_count()-2)
    pool = mp.Pool(use_cores)
    fn = partial(make_single_prediction,builder=specific_builder,loader=loader,model_save_path=model_save_path)

    print('running in parallel on {} processes'.format(pool._processes))
    for (i,(y_p,y,is_disruptive)) in enumerate(pool.imap(fn,shot_list)):
    # for (i,(y_p,y,is_disruptive)) in enumerate(imap(fn,shot_list)):
        print('Shot {}/{}'.format(i,len(shot_list)))
        sys.stdout.flush()
        y_prime.append(y_p)
        y_gold.append(y)
        disruptive.append(is_disruptive)
    pool.close()
    pool.join()
    print('Finished Predictions in {} seconds'.format(time.time()-start_time))
    return y_prime,y_gold,disruptive




def make_single_prediction(shot,specific_builder,loader,model_save_path):
    model = specific_builder.build_model(True)
    model.load_weights(model_save_path)
    model.reset_states()
    X,y = loader.load_as_X_y(shot,prediction_mode=True)
    assert(X.shape[0] == y.shape[0])
    y_p = model.predict(X,batch_size=Loader.get_batch_size(conf['training']['batch_size'],prediction_mode=True),verbose=0)
    answer_dims = y_p.shape[-1]
    if conf['model']['return_sequences']:
        shot_length = y_p.shape[0]*y_p.shape[1]
    else:
        shot_length = y_p.shape[0]
    y_p = np.reshape(y_p,(shot_length,answer_dims))
    y = np.reshape(y,(shot_length,answer_dims))
    is_disruptive = shot.is_disruptive_shot()
    model.reset_states()
    return y_p,y,is_disruptive


def make_predictions_gpu(conf,shot_list,loader):

    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32' #=cpu
    import theano
    from keras.utils.generic_utils import Progbar 
    from plasma.models.builder import ModelBuilder
    specific_builder = ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []

    model = specific_builder.build_model(True)
    specific_builder.load_model_weights(model)
    model.reset_states()

    pbar =  Progbar(len(shot_list))
    shot_sublists = shot_list.sublists(conf['model']['pred_batch_size'],do_shuffle=False,equal_size=True)
    for (i,shot_sublist) in enumerate(shot_sublists):
        X,y,shot_lengths,disr = loader.load_as_X_y_pred(shot_sublist)
        #load data and fit on data
        y_p = model.predict(X,
            batch_size=conf['model']['pred_batch_size'])
        model.reset_states()
        y_p = loader.batch_output_to_array(y_p)
        y = loader.batch_output_to_array(y)
        #cut arrays back
        y_p = [arr[:shot_lengths[j]] for (j,arr) in enumerate(y_p)]
        y = [arr[:shot_lengths[j]] for (j,arr) in enumerate(y)]

        # print('Shots {}/{}'.format(i*num_at_once + j*1.0*len(shot_sublist)/len(X_list),len(shot_list_train)))
        pbar.add(1.0*len(shot_sublist))
        loader.verbose=False#True during the first iteration
        y_prime += y_p
        y_gold += y
        disruptive += disr
    y_prime = y_prime[:len(shot_list)]
    y_gold = y_gold[:len(shot_list)]
    disruptive = disruptive[:len(shot_list)]
    return y_prime,y_gold,disruptive



def make_predictions_and_evaluate_gpu(conf,shot_list,loader):
    y_prime,y_gold,disruptive = make_predictions_gpu(conf,shot_list,loader)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'].loss)
    return roc_area,loss

def make_evaluations_gpu(conf,shot_list,loader):
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32' #=cpu
    import theano
    from keras.utils.generic_utils import Progbar 
    from plasma.models.builder import ModelBuilder
    specific_builder = ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []
    batch_size = min(len(shot_list),conf['model']['pred_batch_size'])

    pbar =  Progbar(len(shot_list))
    print('evaluating {} shots using batchsize {}'.format(len(shot_list),batch_size))

    shot_sublists = shot_list.sublists(batch_size,equal_size=False)
    all_metrics = []
    all_weights = []
    for (i,shot_sublist) in enumerate(shot_sublists):
        batch_size = len(shot_sublist)
        model = specific_builder.build_model(True,custom_batch_size=batch_size)
        specific_builder.load_model_weights(model)
        model.reset_states()
        X,y,shot_lengths,disr = loader.load_as_X_y_pred(shot_sublist,custom_batch_size=batch_size)
        #load data and fit on data
        all_metrics.append(model.evaluate(X,y,batch_size=batch_size,verbose=False))
        all_weights.append(batch_size)
        model.reset_states()

        pbar.add(1.0*len(shot_sublist))
        loader.verbose=False#True during the first iteration

    if len(all_metrics) > 1:
        print('evaluations all: {}'.format(all_metrics))
    loss = np.average(all_metrics,weights = all_weights)
    print('Evaluation Loss: {}'.format(loss))
    return loss 
