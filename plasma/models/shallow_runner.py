from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from itertools import imap

#leading to import errors:
#from hyperopt import hp, STATUS_OK
#from hyperas.distributions import conditional

import time
import sys
import os
from functools import partial
import pathos.multiprocessing as mp

from plasma.conf import conf
from plasma.models.loader import Loader, ProcessGenerator
from plasma.utils.performance import PerformanceAnalyzer
from plasma.utils.evaluation import *
from plasma.utils.state_reset import reset_states

from keras.utils.generic_utils import Progbar 

debug_use_shots = 100000
model_path = "saved_model_new.pkl"
dataset_path = "dataset.npz"
dataset_test_path = "dataset_test.npz"

class FeatureExtractor(object):
    def __init__(self,loader,timesteps = 32):
        self.loader = loader
        self.timesteps = timesteps
        self.positional_fit_order = 4
        self.num_positional_features = self.positional_fit_order + 1 + 3
        self.temporal_fit_order = 3
        self.num_temporal_features = self.temporal_fit_order + 1 + 3


    def load_shots(self,shot_list,sample_prob = 1.0,as_list=False):
        X = []
        Y = []
        Disr = []
        print("loading...")
        pbar =  Progbar(len(shot_list))
        
        fn = partial(self.load_shot,sample_prob=sample_prob)
        pool = mp.Pool()
        print('loading data in parallel on {} processes'.format(pool._processes))
        for x,y,disr in pool.imap(fn,shot_list):
            X.append(x)
            Y.append(y)
            Disr.append(disr)
            pbar.add(1.0)
        pool.close()
        pool.join()
        return X,Y,np.array(Disr)


    def load_shot(self,shot,sample_prob=1.0):
        prepath = self.loader.conf['paths']['processed_prepath']
        save_prepath = prepath + "shallow/"
        save_path = shot.get_save_path(save_prepath)
        if os.path.isfile(save_path):
            dat = np.load(save_path)
            X,Y,disr = dat["X"],dat["Y"],dat["disr"][()]
        else:
            use_signals = self.loader.conf['paths']['use_signals']
            assert(shot.valid)
            shot.restore(prepath)
            if self.loader.normalizer is not None:
                self.loader.normalizer.apply(shot)
            else:
                print('Warning, no normalization. Training data may be poorly conditioned')
            # sig,res = self.get_signal_result_from_shot(shot)
            disr = 1 if shot.is_disruptive else 0
            sig_sample = shot.signals_dict[use_signals[0]] 
            if len(shot.ttd.shape) == 1:
                shot.ttd = np.expand_dims(shot.ttd,axis=1)
            ttd_sample = shot.ttd
            timesteps = self.timesteps
            length = sig_sample.shape[0]
            if length < timesteps:
                print(ttd,shot,shot.number)
                print("Shot must be at least as long as the RNN length.")
                exit(1)
            assert(len(sig_sample.shape) == len(ttd_sample.shape) == 2)
            assert(ttd_sample.shape[1] == 1)

            X = []
            Y = []
            while(len(X) == 0):
                for i in range(length-timesteps+1):
                    if np.random.rand() < sample_prob:
                        x,y = self.get_x_y(i,shot)
                        X.append(x)
                        Y.append(y)
            X = np.stack(X)
            Y = np.stack(Y)
            shot.make_light()
            if not os.path.exists(save_prepath):
                os.makedirs(save_prepath)
            np.savez(save_path,X=X,Y=Y,disr=disr)
            #print(X.shape,Y.shape)
        return X,Y,disr


    def get_x_y(self,timestep,shot):
        x = []
        use_signals = self.loader.conf['paths']['use_signals']
        for sig in use_signals:
            x += [self.extract_features(timestep,shot,sig)]
            # x = sig[timestep:timestep+timesteps,:]
        x = np.concatenate(x,axis=0)
        y = np.round(shot.ttd[timestep+self.timesteps-1,0]).astype(np.int)
        return x,y


    def extract_features(self,timestep,shot,signal):
        raw_sig = shot.signals_dict[signal][timestep:timestep+self.timesteps]
        num_positional_features = self.num_positional_features if signal.num_channels > 1 else 1
        output_arr = np.empty((self.timesteps,num_positional_features))
        final_output_arr = np.empty((num_positional_features*self.num_temporal_features))
        for t in range(self.timesteps):
            output_arr[t,:] = self.extract_positional_features(raw_sig[t,:])
        for i in range(num_positional_features):
            idx = i*self.num_temporal_features
            final_output_arr[idx:idx+self.num_temporal_features] = self.extract_temporal_features(output_arr[:,i])
        return final_output_arr

    def extract_positional_features(self,arr):
        num_channels = len(arr)
        if num_channels > 1:
            ret_arr = np.empty(self.num_positional_features)
            coefficients = np.polynomial.polynomial.polyfit(np.linspace(0,1,num_channels),arr,self.positional_fit_order)
            mu = np.mean(arr)
            std = np.std(arr)
            max_val = np.max(arr)
            ret_arr[:self.positional_fit_order+1] = coefficients
            ret_arr[self.positional_fit_order+1] = mu
            ret_arr[self.positional_fit_order+2] = std
            ret_arr[self.positional_fit_order+3] = max_val
            return ret_arr
        else:
            return arr

    def extract_temporal_features(self,arr):
        ret_arr = np.empty(self.num_temporal_features)
        coefficients = np.polynomial.polynomial.polyfit(np.linspace(0,1,self.timesteps),arr,self.temporal_fit_order)
        mu = np.mean(arr)
        std = np.std(arr)
        max_val = np.max(arr)
        ret_arr[:self.temporal_fit_order+1] = coefficients
        ret_arr[self.temporal_fit_order+1] = mu
        ret_arr[self.temporal_fit_order+2] = std
        ret_arr[self.temporal_fit_order+3] = max_val
        return ret_arr

    def prepend_timesteps(self,arr):
        prepend = arr[0]*np.ones(self.timesteps-1)
        return np.concatenate((prepend,arr))

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,auc,classification_report,confusion_matrix

def train(conf,shot_list_train,shot_list_validate,loader):

    np.random.seed(1)

    print('validate: {} shots, {} disruptive'.format(len(shot_list_validate),shot_list_validate.num_disruptive()))
    print('training: {} shots, {} disruptive'.format(len(shot_list_train),shot_list_train.num_disruptive()))

    feature_extractor = FeatureExtractor(loader)
    shot_list_train = shot_list_train.random_sublist(debug_use_shots)
    X,Y,_ = feature_extractor.load_shots(shot_list_train,sample_prob = 1.0)
    Xv,Yv,_ = feature_extractor.load_shots(shot_list_validate,sample_prob = 1.0)
    X = np.concatenate(X,axis=0)
    Y = np.concatenate(Y,axis=0)
    Xv = np.concatenate(Xv,axis=0)
    Yv = np.concatenate(Yv,axis=0)

    print("Total data: {} samples, {} positive".format(len(X),np.sum(Y > 0)))
    max_samples = 100000
    num_samples = min(max_samples,len(Y))
    indices = np.random.choice(np.array(range(len(Y))),num_samples,replace=False)
    X = X[indices]
    Y = Y[indices]
    
    print("fitting on {} samples, {} positive".format(len(X),np.sum(Y > 0)))

    if not os.path.isfile(model_path):
        
        start_time = time.time()
        #model = svm.SVC(probability=True)
        model = RandomForestClassifier(n_estimators=50,max_depth=20,n_jobs=-1)
        model.fit(X,Y)
        joblib.dump(model,model_path)
        print("Fit model in {} seconds".format(time.time()-start_time))
    else:
        model = joblib.load(model_path)
        print("model exists.")
    

    Y_pred = model.predict(X)
    print("Train")
    print(classification_report(Y,Y_pred))
    Y_predv = model.predict(Xv)
    print("Validate")
    print(classification_report(Yv,Y_predv))
    #print(confusion_matrix(Y,Y_pred))


    print('...done')


def make_predictions(conf,shot_list,loader):
    model = joblib.load(model_path)
    feature_extractor = FeatureExtractor(loader)
    #shot_list = shot_list.random_sublist(10)

    y_prime = []
    y_gold = []
    disruptive = []

    pbar =  Progbar(len(shot_list))
    fn = partial(predict_single_shot,model=model,feature_extractor=feature_extractor)
    pool = mp.Pool()
    print('predicting in parallel on {} processes'.format(pool._processes))
    #for (y_p,y,disr) in map(fn,shot_list):
    for (y_p,y,disr) in pool.imap(fn,shot_list):
        #y_p,y,disr = predict_single_shot(model,feature_extractor,shot)
        y_prime += [np.expand_dims(y_p,axis=1)]
        y_gold += [np.expand_dims(y,axis=1)]
        disruptive += [disr]
        pbar.add(1.0)

    pool.close()
    pool.join()
    return y_prime,y_gold,disruptive

def predict_single_shot(shot,model,feature_extractor):
    X,y,disr = feature_extractor.load_shot(shot,sample_prob=1.0)
    y_p = model.predict_proba(X)[:,1]
    #print(y)
    #print(y_p)
    y = feature_extractor.prepend_timesteps(y)
    y_p = feature_extractor.prepend_timesteps(y_p)
    return y_p,y,disr



def make_predictions_and_evaluate_gpu(conf,shot_list,loader):
    y_prime,y_gold,disruptive = make_predictions(conf,shot_list,loader)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'])
    return y_prime,y_gold,disruptive,roc_area,loss

