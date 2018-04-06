from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from hyperopt import hp, STATUS_OK

import time
import sys
import os
from functools import partial
import pathos.multiprocessing as mp

if sys.version_info[0] < 3:
    from itertools import imap

from plasma.conf import conf
from plasma.models.loader import Loader, ProcessGenerator
from plasma.utils.performance import PerformanceAnalyzer
from plasma.utils.evaluation import *
from plasma.utils.state_reset import reset_states

backend = conf['model']['backend']

def train(conf,shot_list_train,shot_list_validate,loader,shot_list_test=None):
    loader.set_inference_mode(False)
    np.random.seed(1)

    validation_losses = []
    validation_roc = []
    training_losses = []
    print('validate: {} shots, {} disruptive'.format(len(shot_list_validate),shot_list_validate.num_disruptive()))
    print('training: {} shots, {} disruptive'.format(len(shot_list_train),shot_list_train.num_disruptive()))

    if backend == 'tf' or backend == 'tensorflow':
        first_time = "tensorflow" not in sys.modules
        if first_time:
                import tensorflow as tf
                os.environ['KERAS_BACKEND'] = 'tensorflow'
                from keras.backend.tensorflow_backend import set_session
                config = tf.ConfigProto(device_count={"GPU":1})
                set_session(tf.Session(config=config))
    else:
        os.environ['KERAS_BACKEND'] = 'theano'
        os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
        import theano

    from keras.utils.generic_utils import Progbar 
    from keras import backend as K
    from plasma.models import builder

    print('Build model...',end='')
    specific_builder = builder.ModelBuilder(conf)
    train_model = specific_builder.build_model(False) 
    print('Compile model',end='')
    train_model.compile(optimizer=optimizer_class(),loss=conf['data']['target'].loss)
    print('...done')

    #load the latest epoch we did. Returns -1 if none exist yet
    e = specific_builder.load_model_weights(train_model)
    e_start = e
    batch_generator = partial(loader.training_batch_generator_partial_reset,shot_list=shot_list_train)
    batch_iterator = ProcessGenerator(batch_generator())

    num_epochs = conf['training']['num_epochs']
    num_at_once = conf['training']['num_shots_at_once']
    lr_decay = conf['model']['lr_decay']
    print('{} epochs left to go'.format(num_epochs - 1 - e))
    num_so_far_accum = 0
    num_so_far = 0
    num_total = np.inf

    if conf['callbacks']['mode'] == 'max':
        best_so_far = -np.inf
        cmp_fn = max
    else:
        best_so_far = np.inf
        cmp_fn = min

    while e < num_epochs-1:
        e += 1
        print('\nEpoch {}/{}'.format(e+1,num_epochs))
        pbar =  Progbar(len(shot_list_train))

        #decay learning rate each epoch:
        K.set_value(train_model.optimizer.lr, lr*lr_decay**(e))
        
        #print('Learning rate: {}'.format(train_model.optimizer.lr.get_value()))
        num_batches_minimum = 100
        num_batches_current = 0
        training_losses_tmp = []

        while num_so_far < (e - e_start)*num_total or num_batches_current < num_batches_minimum:
            num_so_far_old = num_so_far
            try:
                batch_xs,batch_ys,batches_to_reset,num_so_far_curr,num_total,is_warmup_period = next(batch_iterator)
            except StopIteration:
                print("Resetting batch iterator.")
                num_so_far_accum = num_so_far
                batch_iterator = ProcessGenerator(batch_generator())
                batch_xs,batch_ys,batches_to_reset,num_so_far_curr,num_total,is_warmup_period = next(batch_iterator)
            if np.any(batches_to_reset):
                reset_states(train_model,batches_to_reset)
            if not is_warmup_period:
                num_so_far = num_so_far_accum+num_so_far_curr

                num_batches_current +=1 


                loss = train_model.train_on_batch(batch_xs,batch_ys)
                training_losses_tmp.append(loss)
                pbar.add(num_so_far - num_so_far_old, values=[("train loss", loss)])
                loader.verbose=False#True during the first iteration
            else:
                _ = train_model.predict(batch_xs,batch_size=conf['training']['batch_size'])


        e = e_start+1.0*num_so_far/num_total
        sys.stdout.flush()
        ave_loss = np.mean(training_losses_tmp)
        training_losses.append(ave_loss)
        specific_builder.save_model_weights(train_model,int(round(e)))

        if conf['training']['validation_frac'] > 0.0:
            print("prediction on GPU...")
            _,_,_,roc_area,loss = make_predictions_and_evaluate_gpu(conf,shot_list_validate,loader)
            validation_losses.append(loss)
            validation_roc.append(roc_area)

            epoch_logs = {}
            epoch_logs['val_roc'] = roc_area 
            epoch_logs['val_loss'] = loss
            epoch_logs['train_loss'] = ave_loss
            best_so_far = cmp_fn(epoch_logs[conf['callbacks']['monitor']],best_so_far)
            if best_so_far != epoch_logs[conf['callbacks']['monitor']]: #only save model weights if quantity we are tracking is improving
                print("Not saving model weights")
                specific_builder.delete_model_weights(train_model,int(round(e)))

            if conf['training']['ranking_difficulty_fac'] != 1.0:
                _,_,_,roc_area_train,loss_train = make_predictions_and_evaluate_gpu(conf,shot_list_train,loader)
                batch_iterator.__exit__()
                batch_generator = partial(loader.training_batch_generator_partial_reset,shot_list=shot_list_train)
                batch_iterator = ProcessGenerator(batch_generator())
                num_so_far_accum = num_so_far

        print('=========Summary========')
        print('Training Loss Numpy: {:.3e}'.format(training_losses[-1]))
        if conf['training']['validation_frac'] > 0.0:
            print('Validation Loss: {:.3e}'.format(validation_losses[-1]))
            print('Validation ROC: {:.4f}'.format(validation_roc[-1]))
            if conf['training']['ranking_difficulty_fac'] != 1.0:
                print('Train Loss: {:.3e}'.format(loss_train))
                print('Train ROC: {:.4f}'.format(roc_area_train))
        


    # plot_losses(conf,[training_losses],specific_builder,name='training')
    if conf['training']['validation_frac'] > 0.0:
        plot_losses(conf,[training_losses,validation_losses,validation_roc],specific_builder,name='training_validation_roc')
    batch_iterator.__exit__()
    print('...done')

def optimizer_class():
    from keras.optimizers import SGD,Adam,RMSprop,Nadam,TFOptimizer

    if conf['model']['optimizer'] == 'sgd':
        return SGD(lr=conf['model']['lr'],clipnorm=conf['model']['clipnorm'])
    elif conf['model']['optimizer'] == 'momentum_sgd':
        return SGD(lr=conf['model']['lr'],clipnorm=conf['model']['clipnorm'], decay=1e-6, momentum=0.9)
    elif conf['model']['optimizer'] == 'tf_momentum_sgd':
        return TFOptimizer(tf.train.MomentumOptimizer(learning_rate=conf['model']['lr'],momentum=0.9))
    elif conf['model']['optimizer'] == 'adam':
        return Adam(lr=conf['model']['lr'],clipnorm=conf['model']['clipnorm'])
    elif conf['model']['optimizer'] == 'tf_adam':
        return TFOptimizer(tf.train.AdamOptimizer(learning_rate=conf['model']['lr']))
    elif conf['model']['optimizer'] == 'rmsprop':
        return RMSprop(lr=conf['model']['lr'],clipnorm=conf['model']['clipnorm'])
    elif conf['model']['optimizer'] == 'nadam':
        return Nadam(lr=conf['model']['lr'],clipnorm=conf['model']['clipnorm'])
    else:
        print("Optimizer not implemented yet")
        exit(1)


class HyperRunner(object):
    def __init__(self,conf,loader,shot_list):
        self.loader = loader
        self.shot_list = shot_list
        self.conf = conf

    #FIXME setup for hyperas search
    def keras_fmin_fnct(self,space):
        from plasma.models import builder

        specific_builder = builder.ModelBuilder(self.conf)

        train_model = specific_builder.hyper_build_model(space,False)
        train_model.compile(optimizer=optimizer_class(),loss=conf['data']['target'].loss)

        np.random.seed(1)
        validation_losses = []
        validation_roc = []
        training_losses = []
        shot_list_train,shot_list_validate = self.shot_list.split_direct(1.0-conf['training']['validation_frac'],do_shuffle=True)
        
        from keras.utils.generic_utils import Progbar
        from keras import backend as K

        num_epochs = self.conf['training']['num_epochs']
        num_at_once = self.conf['training']['num_shots_at_once']
        lr_decay = self.conf['model']['lr_decay']

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

            _,_,_,roc_area,loss = make_predictions_and_evaluate_gpu(self.conf,shot_list_validate,self.loader)
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
    loader.set_inference_mode(True)

    use_cores = max(1,mp.cpu_count()-2)

    if backend == 'tf' or backend == 'tensorflow':
        first_time = "tensorflow" not in sys.modules
        if first_time:
                import tensorflow as tf
                os.environ['KERAS_BACKEND'] = 'tensorflow'
                from keras.backend.tensorflow_backend import set_session
                config = tf.ConfigProto(device_count={"CPU":use_cores})
                set_session(tf.Session(config=config))
    else:
        os.environ['THEANO_FLAGS'] = 'device=cpu'
        import theano

    from plasma.models.builder import ModelBuilder
    specific_builder = ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []

    model = specific_builder.build_model(True)
    model.compile(optimizer=optimizer_class(),loss=conf['data']['target'].loss)

    specific_builder.load_model_weights(model)
    model_save_path = specific_builder.get_latest_save_path()

    start_time = time.time()
    pool = mp.Pool(use_cores)
    fn = partial(make_single_prediction,builder=specific_builder,loader=loader,model_save_path=model_save_path)

    print('running in parallel on {} processes'.format(pool._processes))
    for (i,(y_p,y,is_disruptive)) in enumerate(pool.imap(fn,shot_list)):
        print('Shot {}/{}'.format(i,len(shot_list)))
        sys.stdout.flush()
        y_prime.append(y_p)
        y_gold.append(y)
        disruptive.append(is_disruptive)
    pool.close()
    pool.join()
    print('Finished Predictions in {} seconds'.format(time.time()-start_time))
    loader.set_inference_mode(False)
    return y_prime,y_gold,disruptive


def make_single_prediction(shot,specific_builder,loader,model_save_path):
    loader.set_inference_mode(True)
    model = specific_builder.build_model(True)
    model.compile(optimizer=optimizer_class(),loss=conf['data']['target'].loss)

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
    loader.set_inference_mode(False)
    return y_p,y,is_disruptive


def make_predictions_gpu(conf,shot_list,loader,custom_path=None):
    loader.set_inference_mode(True)

    if backend == 'tf' or backend == 'tensorflow':
        first_time = "tensorflow" not in sys.modules
        if first_time:
                import tensorflow as tf
                os.environ['KERAS_BACKEND'] = 'tensorflow'
                from keras.backend.tensorflow_backend import set_session
                config = tf.ConfigProto(device_count={"GPU":1})
                set_session(tf.Session(config=config))
    else:
        os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
        import theano

    from keras.utils.generic_utils import Progbar 
    from plasma.models.builder import ModelBuilder
    specific_builder = ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []

    model = specific_builder.build_model(True)
    model.compile(optimizer=optimizer_class(),loss=conf['data']['target'].loss)

    specific_builder.load_model_weights(model,custom_path)
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

        pbar.add(1.0*len(shot_sublist))
        loader.verbose=False#True during the first iteration
        y_prime += y_p
        y_gold += y
        disruptive += disr
    y_prime = y_prime[:len(shot_list)]
    y_gold = y_gold[:len(shot_list)]
    disruptive = disruptive[:len(shot_list)]
    loader.set_inference_mode(False)
    return y_prime,y_gold,disruptive



def make_predictions_and_evaluate_gpu(conf,shot_list,loader,custom_path=None):
    y_prime,y_gold,disruptive = make_predictions_gpu(conf,shot_list,loader,custom_path)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    shot_list.set_weights(analyzer.get_shot_difficulty(y_prime,y_gold,disruptive))
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'])
    return y_prime,y_gold,disruptive,roc_area,loss

def make_evaluations_gpu(conf,shot_list,loader):
    loader.set_inference_mode(True)

    if backend == 'tf' or backend == 'tensorflow':
        first_time = "tensorflow" not in sys.modules
        if first_time:
                import tensorflow as tf
                os.environ['KERAS_BACKEND'] = 'tensorflow'
                from keras.backend.tensorflow_backend import set_session
                config = tf.ConfigProto(device_count={"GPU":1})
                set_session(tf.Session(config=config))
    else:
        os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
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
        model.compile(optimizer=optimizer_class(),loss=conf['data']['target'].loss)

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
    loader.set_inference_mode(False)
    return loss 
