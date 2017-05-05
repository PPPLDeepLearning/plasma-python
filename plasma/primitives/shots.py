'''
#########################################################
This file containts classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
import os
import os.path
import sys
import random as rnd

import numpy as np

from plasma.utils.processing import train_test_split,cut_and_resample_signal



class ShotListFiles(object):
    def __init__(self,machine,prepath,paths,description=''):
        self.machine = machine
        self.prepath = prepath
        self.paths = paths
        self.description = description

    def __str__(self):
        s = 'machine: ' + self.machine.__str__()
        s += '\n' + self.description
        return s

    def __repr__(self):
        return self.__str__()

    def get_single_shot_numbers_and_disruption_times(self,full_path):
        data = np.loadtxt(full_path,ndmin=1,dtype={'names':('num','disrupt_times'),
                                                                  'formats':('i4','f4')})
        shots = np.array(list(zip(*data))[0])
        disrupt_times = np.array(list(zip(*data))[1])
        return shots, disrupt_times


    def get_shot_numbers_and_disruption_times(self):
        all_shots = []
        all_disruption_times = []
        all_machines_arr = []
        for path in self.paths:
            full_path = self.prepath + path
            shots,disruption_times = self.get_single_shot_numbers_and_disruption_times(full_path)
            all_shots.append(shots)
            all_disruption_times.append(disruption_times)
        return np.concatenate(all_shots),np.concatenate(all_disruption_times)



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

    def load_from_shot_list_files_object(self,shot_list_files_object,signals):
        machine = shot_list_files_object.machine
        shot_numbers,disruption_times = shot_list_files_object.get_shot_numbers_and_disruption_times()
        for number,t in list(zip(shot_numbers,disruption_times)):
            self.append(Shot(number=number,t_disrupt=t,machine=machine,signals=signals))



    def load_from_shot_list_files_objects(self,shot_list_files_objects,signals):
        for obj in shot_list_files_objects:
            self.load_from_shot_list_files_object(obj,signals)

    #         ######Generic Methods####

    # @staticmethod 
    # def get_shots_and_disruption_times(shots_and_disruption_times_path,machine):
    #     data = np.loadtxt(shots_and_disruption_times_path,ndmin=1,dtype={'names':('num','disrupt_times'),
    #                                                               'formats':('i4','f4')})
    #     shots = np.array(list(zip(*data))[0])
    #     disrupt_times = np.array(list(zip(*data))[1])
    #     machines = np.array([machine]*len(shots))
    #     return shots, disrupt_times, machines

    # @staticmethod
    # def get_multiple_shots_and_disruption_times(base_path,endings,machines):
    #     all_shots = []
    #     all_disruption_times = []
    #     all_machines_arr = []
    #     for (ending,machine) in zip(endings,machines):
    #         path = base_path + ending
    #         shots,disruption_times,machines_arr = ShotList.get_shots_and_disruption_times(path,machine)
    #         all_shots.append(shots)
    #         all_disruption_times.append(disruption_times)
    #         all_machines_arr.append(machines_arr)
    #     return np.concatenate(all_shots),np.concatenate(all_disruption_times),np.concatenate(all_machines_arr)


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
            #print('Warning: shot {} not valid, omitting'.format(shot.number))
            return False

        

class Shot(object):
    '''
    A class representing a shot.
    Each shot is a measurement of plasma properties (current, locked mode amplitude, etc.) as a function of time. 

    For 0D data, each shot is modeled as a 2D Numpy array - time vs a plasma property.
    '''

    def __init__(self,number=None,machine=None,signals=None,signals_dict=None,ttd=None,valid=None,is_disruptive=None,t_disrupt=None):
        '''
        Shot objects contain following attributes:
    
         - number: integer, unique identifier of a shot
         - t_disrupt: double, disruption time in milliseconds (second column in the shotlist input file)
         - ttd: Numpy array of doubles, time profile of the shot converted to time-to-disruption values
         - valid: boolean flag indicating whether plasma property (specifically, current) reaches a certain value during the shot
         - is_disruptive: boolean flag indicating whether a shot is disruptive
        '''
        self.number = number #Shot number
        self.machine = machine #machine on which it is defined
        self.signals = signals 
        self.signals_dict = signals_dict #
        self.ttd = ttd 
        self.valid =valid 
        self.is_disruptive = is_disruptive
        self.t_disrupt = t_disrupt
        if t_disrupt is not None:
            self.is_disruptive = Shot.is_disruptive_given_disruption_time(t_disrupt)
        else:
            print('Warning, disruption time (disruptivity) not set! Either set t_disrupt or is_disruptive')

    def __str__(self):
        string = 'number: {}\n'.format(self.number)
        string = 'machine: {}\n'.format(self.machine)
        string += 'signals: {}\n'.format(self.signals )
        string += 'signals_dict: {}\n'.format(self.signals_dict )
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

    def get_data_arrays(self,use_signals):
        t_array = self.ttd
        signal_array = np.zeros((len(t_array),sum([sig.num_channels for sig in use_signals])))
        curr_idx = 0
        for sig in use_signals:
            signal_array[:,curr_idx:curr_idx+sig.num_channels] = self.signals_dict[sig]
            curr_idx += sig.num_channels
        return t_array,signal_array

    def get_individual_signal_arrays(self):
        #guarantee ordering
        return [self.signals_dict[sig] for sig in self.signals]

    def preprocess(self,conf):
        sys.stdout.write('\rrecomputing {}'.format(self.number))
	sys.stdout.flush()
      #get minmax times
        time_arrays,signal_arrays,t_min,t_max,valid = self.get_signals_and_times_from_file(conf) 
        self.valid = valid
        #cut and resample
	if self.valid:
	        self.cut_and_resample_signals(time_arrays,signal_arrays,t_min,t_max,conf)

    def get_signals_and_times_from_file(self,conf):
        valid = True
        t_min = -1
        t_max = np.Inf
        t_thresh = -1
        signal_arrays = []
        time_arrays = []

        #disruptive = self.t_disrupt >= 0

        signal_prepath = conf['paths']['signal_prepath']
        for (i,signal) in enumerate(self.signals):
            t,sig,valid_signal = signal.load_data(signal_prepath,self)
            if not valid_signal:
		return None,None,None,None,False
            else:
            	assert(len(sig.shape) == 2)
            	assert(len(t.shape) == 1)
		assert(len(t) > 1)
                t_min = max(t_min,t[0])
                t_max = min(t_max,t[-1])
                signal_arrays.append(sig)
                time_arrays.append(t)
        assert(t_max > t_min or not valid)
	#make sure the shot is long enough.
	dt = conf['data']['dt']
	if (t_max - t_min)/dt <= (conf['model']['length']+conf['data']['T_min_warn']):
	    print('Shot {} contains insufficient data'.format(self.number))
	    valid = False
		
	
        if self.is_disruptive:
            t_max = self.t_disrupt
            assert(self.t_disrupt <= t_max or not valid)

        return time_arrays,signal_arrays,t_min,t_max,valid


    def cut_and_resample_signals(self,time_arrays,signal_arrays,t_min,t_max,conf):
        dt = conf['data']['dt']
        signals_dict = dict()

        #resample signals
        assert((len(signal_arrays) == len(time_arrays) == len(self.signals)) and len(signal_arrays) > 0)
        tr = 0
        for (i,signal) in enumerate(self.signals):
            tr,sigr = cut_and_resample_signal(time_arrays[i],signal_arrays[i],t_min,t_max,dt)
            signals_dict[signal] = sigr

        ttd = self.convert_to_ttd(tr,conf)
        self.signals_dict =signals_dict 
        self.ttd = ttd

    def convert_to_ttd(self,tr,conf):
        T_max = conf['data']['T_max']
	dt = conf['data']['dt']
        if self.is_disruptive:
            ttd = max(tr) - tr
            ttd = np.clip(ttd,0,T_max)
        else:
            ttd = T_max*np.ones_like(tr)
        ttd = np.log10(ttd + 1.0*dt/10)
        return ttd

    def save(self,prepath):
        if not os.path.exists(prepath):
            os.makedirs(prepath)
        save_path = self.get_save_path(prepath)
        np.savez(save_path,valid=self.valid,is_disruptive=self.is_disruptive,
            signals_dict=self.signals_dict,ttd=self.ttd)
        print('...saved shot {}'.format(self.number))

    def get_save_path(self,prepath):
        return get_individual_shot_file(prepath,self.number,'.npz')

    def restore(self,prepath,light=False):
        assert self.previously_saved(prepath), 'shot was never saved'
        save_path = self.get_save_path(prepath)
        dat = np.load(save_path)

        self.valid = dat['valid'][()]
        self.is_disruptive = dat['is_disruptive'][()]

        if light:
            self.signals_dict = None
            self.ttd = None 
        else:
            self.signals_dict = dat['signals_dict'][()]
            self.ttd = dat['ttd']
  
    def previously_saved(self,prepath):
        save_path = self.get_save_path(prepath)
        return os.path.isfile(save_path)

    def make_light(self):
        self.signals_dict = None
        self.ttd = None

    @staticmethod
    def is_disruptive_given_disruption_time(t):
        return t >= 0

#it used to be in utilities, but can't import globals in multiprocessing
def get_individual_shot_file(prepath,shot_num,ext='.txt'):
    return prepath + str(shot_num) + ext 
