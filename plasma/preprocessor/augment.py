from __future__ import print_function
import os
import time,sys
import abc
import numpy as np

from plasma.primitives.shots import ShotList, Shot

class AbstractAugmentator(object):

    def __init__(self,normalizer,is_inference,conf):
        self.conf = conf
        self.to_augment = self.conf['data']['signals_to_augment']
        self.normalizer = normalizer
        self.is_inference = is_inference

    #set whether we are training or testing
    def set_inference(self,is_inference):
        self.is_inference = is_inference

    def __str__(self):
        s = self.normalizer.__str__()
        s += "\nIs being augmented!".format(self.to_augment)
        s += "Signal to augmented: {}\n".format(self.to_augment)
        s += "Is inference: {}\n".format(self.is_inference)
        return s

    @abc.abstractmethod
    def apply(self,shot):
        pass

    @abc.abstractmethod
    def augment(self,sig):
        pass

class Augmentator(AbstractAugmentator):

    def apply(self,shot):
        #first just apply normalization as usual.
        self.normalizer.apply(shot)
        #during inference, augment a specific signal
        if self.is_inference:
            to_augment = self.to_augment
        else: 
            #during training augment a random signal
            if self.conf['data']['augment_during_training']:
                to_augment =np.random.sample([x.description for x in shot.signals])
            else:
                to_augment = None 
        if to_augment is not None:
            #FIXME 
            for (i,sig) in enumerate(shot.signals):
                if sig.description in self.conf['data']['signals_to_augment']:
                    print ('Augmenting {} signal'.format(sig.description))
                    shot.signals_dict[sig] = self.augment(shot.signals_dict[sig])

    def augment(self,signal,strength=10):
        '''
        The purpose of the method is to modify a signal specified by a configuration parameter or at random according to 
        a specific mode. Modes include: noise, zeroing and no augmentation.

        It performs calls to: numpy random number generator

        Argument list: 
          - signal: signal
          - strength: strength of the noise, measured in standard deviations. Integer, default value: 10

        Config parameters list:
          - conf['data']['augmentation_mode']: categorical config parameter specifying how to augment. Possible values
                                               include "noise", "zero" and "none" (strings)

        Returns:  
          - signal: augmented signal ... numpy array of numeric types?
        '''
        if self.conf['data']['augmentation_mode'] == "noise":
            return signal + np.random.normal(0,strength,signal.shape) 
        elif self.conf['data']['augmentation_mode'] == "zero":
            return signal*0.0 #if "set to zero" augmentation. Can control in conf.
        elif self.conf['data']['augmentation_mode'] == "none":
            return signal #if no augmentation. Should be the default in conf.
        else:
            print("Unknown augmentation mode. Exiting")
            exit(-1)
