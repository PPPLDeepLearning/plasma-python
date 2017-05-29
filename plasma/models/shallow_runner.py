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

backend = conf['model']['backend']


def FeatureExtractor(object):
	def __init__(self,loader,timesteps = 32):
		self.loader = loader
		self.timesteps = timesteps
		self.positional_fit_order = 4
		self.num_positional_features = self.positional_fit_order + 1 + 3
		self.temporal_fit_order = 3
		self.num_temporal_features = self.temporal_fit_order + 1 + 3


	def load_shots(self,shot_list):
		X = []
		Y = []
		for shot in shot_list:
			x,y = self.load_shot(shot)
			X.append(x)
			Y.append(y)
		return np.vstack(X),np.vstack(Y)


	def load_shot(self,shot):
		prepath = self.loader.conf['paths']['processed_prepath']
        use_signals = self.loader.conf['paths']['use_signals']
        assert(isinstance(shot,Shot))
        assert(shot.valid)
        shot.restore(prepath)
        if self.loader.normalizer is not None:
            self.loader.normalizer.apply(shot)
        else:
            print('Warning, no normalization. Training data may be poorly conditioned')
		# sig,res = self.get_signal_result_from_shot(shot)
		sig_sample = shot.signals_dict[use_signals[0]] 
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
		for i in range(length-timesteps+1)
			x,y = get_x_y(self,i,shot)
			X.append(x)
			Y.append(y)
		X = np.stack(X)
		Y = np.stack(Y)

        shot.make_light()
        return X,Y


    def get_x_y(self,timestep,shot):
    	x = []
		for sig in use_signals
			x += [self.extract_features(timestep,shot,sig)]
			# x = sig[timestep:timestep+timesteps,:]
		x = np.concatenate(x,axis=0)
		y = np.round(res[timestep+timesteps-1,0]).astype(np.int)


    def extract_features(self,timestep,shot,signal):
    	raw_sig = shot.signals_dict[signal][timestep:timestep+self.timesteps]
    	num_positional_features = self.num_positional_features if signal.num_channels > 1 else 1
	    output_arr = np.empty((self.timesteps,num_positional_features))
	    final_output_arr = np.empty((num_positional_features*self.num_temporal_features))
    	for t in range(self.timesteps):
    		output_arr[t,:] = self.extract_positional_features(raw_sig[t,:])
    	for i in range(num_positional_features)
    		idx = i*self.num_temporal_features
    		final_output_arr[idx:idx+self.num_temporal_features] = self.extract_temporal_features(output_arr[:,i])
    	return final_output_arr

    def extract_positional_features(self,arr):
    	num_channels = len(arr)
    	if num_channels > 1
	    	ret_arr = np.empty(self.num_positional_features)
	    	coefficients,_ = np.polynomial.polynomial.polyfit(np.linspace(0,1,signal.num_channels),arr,self.positional_fit_order)
	    	mu = np.mean(arr)
	    	std = np.std(arr)
	    	max_val = np.max(arr)
	    	ret_arr[:positional_fit_order+1] = coefficients
	    	ret_arr[positional_fit_order+1] = mu
	    	ret_arr[positional_fit_order+2] = std
	    	ret_arr[positional_fit_order+3] = max_val
    		return ret_arr
    	else:
    		return arr

	def extract_temporal_features(self,arr):
	    ret_arr = np.empty(self.num_temporal_features)
	    coefficients,_ = np.polynomial.polynomial.polyfit(np.linspace(0,1,self.timesteps),arr,self.temporal_fit_order)
    	mu = np.mean(arr)
    	std = np.std(arr)
    	max_val = np.max(arr)
    	ret_arr[:temporal_fit_order+1] = coefficients
    	ret_arr[temporal_fit_order+1] = mu
    	ret_arr[temporal_fit_order+2] = std
    	ret_arr[temporal_fit_order+3] = max_val
		return ret_arr

from sklearn import svm
from sklearn.externals import joblib

def train(conf,shot_list_train,shot_list_validate,loader):

    np.random.seed(1)

    print('validate: {} shots, {} disruptive'.format(len(shot_list_validate),shot_list_validate.num_disruptive()))
    print('training: {} shots, {} disruptive'.format(len(shot_list_train),shot_list_train.num_disruptive()))


    feature_extractor = FeatureExtractor(loader)
    X,Y = feature_extractor.load_shots(shot_list_train)

    model = svm.SVC()
    model.fit(X,Y)

    joblib.dump(model,'saved_model.pkl')

    print('...done')


def make_predictions(conf,shot_list,loader):
	model = joblib.load('saved_model.pkl')
    feature_extractor = FeatureExtractor(loader)

    y_prime = []
    y_gold = []
    disruptive = []

    from keras.utils.generic_utils import Progbar 
    pbar =  Progbar(len(shot_list))
    for shot in shot_list:
    	X,Y = feature_extractor.load_shot(shot)
    	Y_pred = model.predict(X)
    	disr = 1 if shot.is_disruptive else 0

        y_prime += [y_p]
        y_gold += [y]
        disruptive += [disr]
        pbar.add(1.0)
    return y_prime,y_gold,disruptive



def make_predictions_and_evaluate_gpu(conf,shot_list,loader):
    y_prime,y_gold,disruptive = make_predictions(conf,shot_list,loader)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'])
    return y_prime,y_gold,disruptive,roc_area,loss

