'''
#########################################################
This file containts classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
import os
import time,sys
import abc

import numpy as np
from scipy.signal import exponential,correlate
import pathos.multiprocessing as mp

from plasma.primitives.shots import ShotList, Shot
from plasma.utils.processing import get_signal_slices

#######NORMALIZATION##########
class Stats(object):
    pass


class Normalizer(object):
    def __init__(self,conf):
        self.num_processed = 0
        self.num_disruptive = 0
        self.conf = conf
        self.path = conf['paths']['normalizer_path']
        self.remapper = conf['data']['target'].remapper


    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def extract_stats(self,shot):
        pass

    @abc.abstractmethod
    def incorporate_stats(self,stats):
        pass

    @abc.abstractmethod
    def apply(self,shot):
        pass

    @abc.abstractmethod
    def save_stats(self):
        pass

    @abc.abstractmethod
    def load_stats(self):
        pass

    ######Modify the above to change the specifics of the normalization scheme#######

    def train(self):
        conf = self.conf
        #only use training shots here!! "Don't touch testing shots"
        shot_files = conf['paths']['shot_files']# + conf['paths']['shot_files_test']
        # shot_list_dir = conf['paths']['shot_list_dir']
        use_shots = max(400,conf['data']['use_shots'])
        return self.train_on_files(shot_list_dir,shot_files,use_shots)


    def train_on_files(self,shot_files,use_shots):
        conf = self.conf
        all_signals = conf['paths']['all_signals'] 
	shot_list = ShotList()
        shot_list.load_from_shot_list_files_objects(shot_files,all_signals)
        shot_list_picked = shot_list.random_sublist(use_shots) 

        recompute = conf['data']['recompute_normalization']

        if recompute or not self.previously_saved_stats():
            use_cores = max(1,mp.cpu_count()-2)
            pool = mp.Pool(use_cores)
            print('running in parallel on {} processes'.format(pool._processes))
            start_time = time.time()

            for (i,stats) in enumerate(pool.imap_unordered(self.train_on_single_shot,shot_list_picked)):
                self.incorporate_stats(stats)
                sys.stdout.write('\r' + '{}/{}'.format(i,len(shot_list_picked)))

            pool.close()
            pool.join()
            print('Finished Training Normalizer on {} files in {} seconds'.format(len(shot_list_picked),time.time()-start_time))
            self.save_stats()
        else:
            self.load_stats()
        print(self)


    def cut_end_of_shot(self,shot):
        T_min_warn = self.conf['data']['T_min_warn']
        for key in shot.signals_dict:
            shot.signals_dict[key] = shot.signals_dict[key][:-T_min_warn,:]
        shot.ttd = shot.ttd[:-T_min_warn]

    # def apply_mask(self,shot):
    #     use_signals = self.conf['paths']['use_signals']
    #     return shot.get_data_arrays(use_signals)

    # def apply_positivity_mask(self,shot):
    #     mask = self.conf['paths']['positivity_mask']
    #     mask = [np.array(subl) for subl in mask]
    #     indices = np.concatenate([indices_sublist[mask[i]] for i,indices_sublist in enumerate(self.get_indices_list())])
    #     shot.signals[:,indices] = np.clip(shot.signals[:,indices],0,np.Inf)

    def train_on_single_shot(self,shot):
        assert isinstance(shot,Shot), 'should be instance of shot'
        processed_prepath = self.conf['paths']['processed_prepath']
        shot.restore(processed_prepath)
        stats = self.extract_stats(shot) 
        shot.make_light()
        return stats


    def previously_saved_stats(self):
        return os.path.isfile(self.path)

    # def get_indices_list(self):
    #     return get_signal_slices(self.conf['paths']['signals_dirs'])





class MeanVarNormalizer(Normalizer):
    def __init__(self,conf):
        Normalizer.__init__(self,conf)
        self.means = None
        self.stds = None

    def __str__(self):
        means = np.median(self.means,axis=0)
        stds = np.median(self.stds,axis=0)
        return('Mean Var Normalizer.\nmeans: {}\nstds: {}'.format(means,stds))

    def extract_stats(self,shot):
        stats = Stats()
        if shot.valid:
            list_of_signals = shot.get_individual_signal_arrays()
            num_signals = len(list_of_signals)
            stats.means = np.reshape(np.array([np.mean(sig) for sig in list_of_signals]),(1,num_signals))
            stats.stds = np.reshape(np.array([np.std(sig) for sig in list_of_signals]),(1,num_signals))
            stats.is_disruptive = shot.is_disruptive
        else:
            print('Warning: shot {} not valid, omitting'.format(shot.number))
        stats.valid = shot.valid
        return stats


    def incorporate_stats(self,stats):
        if stats.valid:
            means = stats.means
            stds = stats.stds
            if self.num_processed == 0:
                self.means = means
                self.stds = stds 
            else:
                self.means = np.concatenate((self.means,means),axis=0)
                self.stds = np.concatenate((self.stds,stds),axis=0)
            self.num_processed = self.num_processed + 1
            self.num_disruptive = self.num_disruptive + (1 if stats.is_disruptive else 0)


    def apply(self,shot):
        assert self.means is not None and self.stds is not None, "self.means or self.stds not initialized"
        means = np.median(self.means,axis=0)
        stds = np.median(self.stds,axis=0)
        for (i,sig) in enumerate(shot.signals):
            shot.signals_dict[sig] = (shot.signals_dict[sig] - means[i])/stds[i]
        shot.ttd = self.remapper(shot.ttd,self.conf['data']['T_warning'])
        self.cut_end_of_shot(shot)
        # self.apply_positivity_mask(shot)
        # self.apply_mask(shot)


    def save_stats(self):
        # standard_deviations = dat['standard_deviations']
        # num_processed = dat['num_processed']
        # num_disruptive = dat['num_disruptive']
        np.savez(self.path,means = self.means,stds = self.stds,
         num_processed=self.num_processed,num_disruptive=self.num_disruptive)
        print('saved normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))

    def load_stats(self):
        assert self.previously_saved_stats(), "stats not saved before"
        dat = np.load(self.path)
        self.means = dat['means']
        self.stds = dat['stds']
        self.num_processed = dat['num_processed']
        self.num_disruptive = dat['num_disruptive']
        print('loaded normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))
        #print('loading normalization data from {} shots, {} disruptive'.format(num_processed,num_disruptive))


class VarNormalizer(MeanVarNormalizer):
    def apply(self,shot):
        assert self.means is not None and self.stds is not None, "self.means or self.stds not initialized"
        stds = np.median(self.stds,axis=0)
        for (i,sig) in enumerate(shot.signals):
            shot.signals_dict[sig] = (shot.signals_dict[sig])/stds[i]
        shot.ttd = self.remapper(shot.ttd,self.conf['data']['T_warning'])
        self.cut_end_of_shot(shot)

    def __str__(self):
        stds = np.median(self.stds,axis=0)
        return('Var Normalizer.\nstds: {}'.format(stds))


class AveragingVarNormalizer(VarNormalizer):

    def apply(self,shot):
        super(AveragingVarNormalizer,self).apply(shot)
        window_decay = self.conf['data']['window_decay']
        window_size = self.conf['data']['window_size']
        window = exponential(window_size,0,window_decay,False)
        window /= np.sum(window)
        for (i,sig) in enumerate(shot.signals):
            shot.signals_dict[sig] = apply_along_axis(lambda m : correlate(m,window,'valid'),axis=0,arr=shot.signals_dict[sig])
        shot.ttd = shot.ttd[-shot.signals.shape[0]:]

    def __str__(self):
        window_decay = self.conf['data']['window_decay']
        window_size = self.conf['data']['window_size']
        stds = np.median(self.stds,axis=0)
        return('Averaging Var Normalizer.\nstds: {}\nWindow size: {}, Window decay: {}'.format(stds,window_size,window_decay))


class MinMaxNormalizer(Normalizer):
    def __init__(self,conf):
        Normalizer.__init__(self,conf)
        self.minimums = None
        self.maximums = None



    def __str__(self):
        return('Normalizer.\nminimums: {}\nmaximums: {}'.format(self.minimums,self.maximums))

    def extract_stats(self,shot):
        stats = Stats()
        if shot.valid:
            list_of_signals = shot.get_individual_signal_arrays()
            stats.minimums = np.array([np.min(sig) for sig in list_of_signals])
            stats.maximums = np.array([np.max(sig) for sig in list_of_signals])
            stats.is_disruptive = shot.is_disruptive
        else:
            print('Warning: shot {} not valid, omitting'.format(shot.number))
        stats.valid = shot.valid
        return stats


    def incorporate_stats(self,stats):
        if stats.valid:
            minimums = stats.minimums
            maximums = stats.maximums
            if self.num_processed == 0:
                self.minimums = minimums
                self.maximums = maximums
            else:
                self.minimums = (self.num_processed*self.minimums + minimums)/(self.num_processed + 1.0)#snp.min(vstack((self.minimums,minimums)),0)
                self.maximums = (self.num_processed*self.maximums + maximums)/(self.num_processed + 1.0)#snp.max(vstack((self.maximums,maximums)),0)
            self.num_processed = self.num_processed + 1
            self.num_disruptive = self.num_disruptive + (1 if stats.is_disruptive else 0)


    def apply(self,shot):
        assert(self.minimums is not None and self.maximums is not None) 
        shot.signals = (shot.signals - self.minimums)/(self.maximums - self.minimums)
        for (i,sig) in enumerate(shot.signals):
            shot.signals_dict[sig] = (shot.signals_dict[sig] - self.minimums)/(self.maximums - self.minimums)
        shot.ttd = self.remapper(shot.ttd,self.conf['data']['T_warning'])
        self.cut_end_of_shot(shot)
        # self.apply_positivity_mask(shot)
        # self.apply_mask(shot)

    def save_stats(self):
        # standard_deviations = dat['standard_deviations']
        # num_processed = dat['num_processed']
        # num_disruptive = dat['num_disruptive']
        np.savez(self.path,minimums = self.minimums,maximums = self.maximums,
         num_processed=self.num_processed,num_disruptive=self.num_disruptive)
        print('saved normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))

    def load_stats(self):
        assert(self.previously_saved_stats())
        dat = np.load(self.path)
        self.minimums = dat['minimums']
        self.maximums = dat['maximums']
        self.num_processed = dat['num_processed']
        self.num_disruptive = dat['num_disruptive']
        print('loaded normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))
        #print('loading normalization data from {} shots, {} disruptive'.format(num_processed,num_disruptive))


def get_individual_shot_file(prepath,shot_num,ext='.txt'):
    return prepath + str(shot_num) + ext 
