'''
#########################################################
This file containts classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
import os
import random as rnd

import numpy as np

from plasma.utils.processing import train_test_split

class ShotList(object):
    '''
    A wrapper class around list of Shot objects, providing utilities to 
    extract, load and transform Shots before passing them to an estimator.

    During distributed training, shot lists are split into sublists.
    A sublist is a ShotList object having num_at_once shots. The ShotList contains an entire dataset 
    as specified in the configuration file.
    '''

    def __init__(self,shots=None):
        '''
        A ShotList is a list of 2D Numpy arrays.
        '''
        self.shots = []
        if shots is not None:
            assert(all([isinstance(shot,Shot) for shot in shots]))
            self.shots = [shot for shot in shots]

    def load_from_files(self,shot_list_dir,shot_files):
        shot_numbers,disruption_times = ShotList.get_multiple_shots_and_disruption_times(shot_list_dir,shot_files)
        for number,t in zip(shot_numbers,disruption_times):
            self.append(Shot(number=number,t_disrupt=t))

            ######Generic Methods####

    @staticmethod 
    def get_shots_and_disruption_times(shots_and_disruption_times_path):
        data = np.loadtxt(shots_and_disruption_times_path,ndmin=1,dtype={'names':('num','disrupt_times'),
                                                                  'formats':('i4','f4')})
        shots = np.array(zip(*data)[0])
        disrupt_times = np.array(zip(*data)[1])
        return shots, disrupt_times

    @staticmethod
    def get_multiple_shots_and_disruption_times(base_path,endings):
        all_shots = []
        all_disruption_times = []
        for ending in endings:
            path = base_path + ending
            shots,disruption_times = ShotList.get_shots_and_disruption_times(path)
            all_shots.append(shots)
            all_disruption_times.append(disruption_times)
        return np.concatenate(all_shots),np.concatenate(all_disruption_times)


    def split_train_test(self,conf):
        shot_list_dir = conf['paths']['shot_list_dir']
        shot_files = conf['paths']['shot_files']
        shot_files_test = conf['paths']['shot_files_test']
        train_frac = conf['training']['train_frac']
        shuffle_training = conf['training']['shuffle_training']
        use_shots = conf['data']['use_shots']
        #split randomly
        use_shots_train = int(round(train_frac*use_shots))
        use_shots_test = int(round((1-train_frac)*use_shots))
        if len(shot_files_test) == 0:
            shot_list_train,shot_list_test = train_test_split(self.shots,train_frac,shuffle_training)
    	    shot_numbers_train = [shot.number for shot in shot_list_train]
    	    shot_numbers_test = [shot.number for shot in shot_list_test]
        #train and test list given
        else:
            shot_numbers_train,_ = ShotList.get_multiple_shots_and_disruption_times(shot_list_dir,shot_files)
            shot_numbers_test,_ = ShotList.get_multiple_shots_and_disruption_times(shot_list_dir,shot_files_test)

        
    	print(len(shot_numbers_train),len(shot_numbers_test))
        shots_train = self.filter_by_number(shot_numbers_train)
        shots_test = self.filter_by_number(shot_numbers_test)
        return shots_train.random_sublist(use_shots_train),shots_test.random_sublist(use_shots_test)


    def split_direct(self,frac,do_shuffle=True):
        shot_list_one,shot_list_two = train_test_split(self.shots,frac,do_shuffle)
        return ShotList(shot_list_one),ShotList(shot_list_two)



    def filter_by_number(self,numbers):
        new_shot_list = ShotList()
        numbers = set(numbers)
        for shot in self.shots:
            if shot.number in numbers:
                new_shot_list.append(shot)
        return new_shot_list

    def num_disruptive(self):
        return len([shot for shot in self.shots if shot.is_disruptive_shot()])

    def __len__(self):
        return len(self.shots) 

    def __str__(self):
        return str([s.number for s in self.shots])

    def __iter__(self):
        return self.shots.__iter__()

    def next(self):
        return self.__iter__().next()

    def __add__(self,other_list):
        return ShotList(self.shots + other_list.shots)


    def random_sublist(self,num):
        num = min(num,len(self))
        shots_picked = np.random.choice(self.shots,size=num,replace=False)
        return ShotList(shots_picked)

    def sublists(self,num,do_shuffle=True,equal_size=False):
        lists = []
        if do_shuffle:
            self.shuffle()
        for i in range(0,len(self),num):
            subl = self.shots[i:i+num]
            while equal_size and len(subl) < num:
                subl.append(rnd.choice(self.shots))
            lists.append(subl)
        return [ShotList(l) for l in lists]



    def shuffle(self):
        np.random.shuffle(self.shots)

    def as_list(self):
        return self.shots

    def append(self,shot):
        assert(isinstance(shot,Shot))
        self.shots.append(shot)

    def make_light(self):
        for shot in self.shots:
            shot.make_light()

    def append_if_valid(self,shot):
        if shot.valid:
            self.append(shot)
            return True
        else:
            print('Warning: shot {} not valid, omitting'.format(shot.number))
            return False

        

class Shot(object):
    '''
    A class representing a shot.
    Each shot is a measurement of plasma properties (current, locked mode amplitude, etc.) as a function of time. 

    For 0D data, each shot is modeled as a 2D Numpy array - time vs a plasma property.
    '''

    def __init__(self,number=None,signals=None,ttd=None,valid=None,is_disruptive=None,t_disrupt=None):
        '''
        Shot objects contain following attributes:
    
         - number: integer, unique identifier of a shot
         - t_disrupt: double, disruption time in milliseconds (second column in the shotlist input file)
         - ttd: Numpy array of doubles, time profile of the shot converted to time-to-disruption values
         - valid: boolean flag indicating whether plasma property (specifically, current) reaches a certain value during the shot
         - is_disruptive: boolean flag indicating whether a shot is disruptive
        '''
        self.number = number #Shot number
        self.signals = signals 
        self.ttd = ttd 
        self.valid =valid 
        self.is_disruptive = is_disruptive
        self.t_disrupt = t_disrupt
        if t_disrupt is not None:
            self.is_disruptive = Shot.is_disruptive_given_disruption_time(t_disrupt)

    def __str__(self):
        string = 'number: {}\n'.format(self.number)
        string += 'signals: {}\n'.format(self.signals )
        string += 'ttd: {}\n'.format(self.ttd )
        string += 'valid: {}\n'.format(self.valid )
        string += 'is_disruptive: {}\n'.format(self.is_disruptive)
        string += 't_disrupt: {}\n'.format(self.t_disrupt)
        return string
     

    def get_number(self):
        return self.number

    def get_signals(self):
        return self.signals

    def is_valid(self):
        return self.valid

    def is_disruptive_shot(self):
        return self.is_disruptive

    def save(self,prepath):
        if not os.path.exists(prepath):
            os.makedirs(prepath)
        save_path = self.get_save_path(prepath)
        np.savez(save_path,number=self.number,valid=self.valid,is_disruptive=self.is_disruptive,
            signals=self.signals,ttd=self.ttd)
        print('...saved shot {}'.format(self.number))

    def get_save_path(self,prepath):
        return get_individual_shot_file(prepath,self.number,'.npz')

    def restore(self,prepath,light=False):
        assert self.previously_saved(prepath), 'shot was never saved'
        save_path = self.get_save_path(prepath)
        dat = np.load(save_path)

        self.number = dat['number'][()]
        self.valid = dat['valid'][()]
        self.is_disruptive = dat['is_disruptive'][()]

        if light:
            self.signals = None
            self.ttd = None 
        else:
            self.signals = dat['signals']
            self.ttd = dat['ttd']
  
    def previously_saved(self,prepath):
        save_path = self.get_save_path(prepath)
        return os.path.isfile(save_path)

    def make_light(self):
        self.signals = None
        self.ttd = None

    @staticmethod
    def is_disruptive_given_disruption_time(t):
        return t >= 0

#it used to be in utilities, but can't import globals in multiprocessing
def get_individual_shot_file(prepath,shot_num,ext='.txt'):
    return prepath + str(shot_num) + ext 
