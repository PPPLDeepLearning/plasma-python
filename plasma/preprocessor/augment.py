from __future__ import print_function
import abc
import numpy as np
import random


class ByShotAugmentator(object):
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __str__(self):
        s = self.normalizer.__str__()
        s += "\n including by shot augmentation"
        return s

    def apply(self, shot):
        '''
        The purpose of the method is to apply normalization to a shot and then
        optionally apply augmentation with a function that is individual to
        every shot.

        Argument list:
          - shot: plasma shot. Should contain an augment function

        Config parameters list:
          - conf['data']['augment_during_training']: boolean flag, yes or no to
        augment during training

        '''
        # first just apply normalization as usual.
        self.normalizer.apply(shot)
        if shot.augmentation_fn is not None:
            shot.augmentation_fn(shot)

    def set_inference_mode(self, is_inference):
        self.normalizer.set_inference_mode(is_inference)


class AbstractAugmentator(object):

    def __init__(self, normalizer, is_inference, conf):
        self.conf = conf
        self.to_augment_str = self.conf['data']['signal_to_augment']
        self.normalizer = normalizer
        self.is_inference = is_inference

    # set whether we are training or testing
    def set_inference(self, is_inference):
        self.is_inference = is_inference

    def __str__(self):
        s = self.normalizer.__str__()
        s += "\nIs being augmented!".format(self.to_augment_str)
        s += "Signal to augmented: {}\n".format(self.to_augment_str)
        s += "Is inference: {}\n".format(self.is_inference)
        return s

    # for compatibility with code that changes the mode of the normalizer
    def set_inference_mode(self, is_inference):
        self.normalizer.set_inference_mode(is_inference)

    @abc.abstractmethod
    def apply(self, shot):
        pass

    @abc.abstractmethod
    def augment(self, sig):
        pass


class Augmentator(AbstractAugmentator):

    def apply(self, shot):
        '''
        The purpose of the method is to apply normalization to a shot and then
        optionally apply augmentation.  During inference, a specific signal
        (one at a time) is augmented based on the string supplied in the config
        file.  During training, augment a random signal (again, one at a time)
        or do not augment at all.

        It performs calls to: Augmentator.augment(), random.random.choice

        Argument list:
          - shot: plasma shot

        Config parameters list:
          - conf['data']['augment_during_training']: boolean flag, yes or no to
        augment during training

        '''
        # first just apply normalization as usual.
        self.normalizer.apply(shot)
        if self.is_inference:
            # during inference, augment a specific signal (one at a time)
            to_augment_str = self.to_augment_str
        else:
            # during training augment a random signal, one at a time
            if self.conf['data']['augment_during_training']:
                to_augment_str = random.choice(
                    [x.description for x in shot.signals])
            else:
                to_augment_str = None
        if to_augment_str is not None:
            # FIXME might be better to use search. are we always going to
            # augment 1 signal at a time?
            for (i, sig) in enumerate(shot.signals):
                if sig.description == to_augment_str:
                    print('Augmenting {} signal'.format(sig.description))
                    shot.signals_dict[sig] = self.augment(
                        shot.signals_dict[sig])

    def augment(self, signal, strength=10):
        '''
        The purpose of the method is to modify a signal specified by a
        configuration parameter or at random according to a specific
        mode. Modes include: noise, zeroing and no augmentation.

        It performs calls to: numpy random number generator

        Argument list:
          - signal: signal
          - strength: strength of the noise, measured in standard
        deviations. Integer, default value: 10

        Config parameters list:
          - conf['data']['augmentation_mode']: categorical config parameter
        specifying how to augment. Possible values include "noise", "zero" and
        "none" (strings)

        Returns:
          - signal: augmented signal ... numpy array of numeric types?
        '''
        if self.conf['data']['augmentation_mode'] == "noise":
            return np.random.normal(0, strength, signal.shape)
        elif self.conf['data']['augmentation_mode'] == "zero":
            # if "set to zero" augmentation. Can control in conf.
            return signal*0.0
        elif self.conf['data']['augmentation_mode'] is None:
            # no augmentation should be the default in conf.yaml
            return signal
        else:
            print("Unknown augmentation mode. Exiting")
            exit(-1)
