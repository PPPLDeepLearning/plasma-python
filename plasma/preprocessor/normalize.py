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

'''TODO
- incorporate stats, pass machine (perhaps save machine in stats object!)
- incorporate stats, have a dictionary of aggregate stats for every machine.
- check "is_previously_saved" by making sure there is a normalizer for every machine
'''




#######NORMALIZATION##########
class Stats(object):
    pass


class Normalizer(object):
    def __init__(self,conf):
        self.num_processed = dict()
        self.num_disruptive = dict()
        self.conf = conf
        self.path = conf['paths']['normalizer_path']
        self.remapper = conf['data']['target'].remapper
        self.machines = set()

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

    def ensure_machine(self,machine):
        if machine not in self.means:
                self.num_processed[machine] = 0
                self.num_disruptive[machine] = 0
    ######Modify the above to change the specifics of the normalization scheme#######

    def train(self):
        conf = self.conf
        #only use training shots here!! "Don't touch testing shots"
        shot_files = conf['paths']['shot_files']# + conf['paths']['shot_files_test']
        shot_files_all = conf['paths']['shot_files_all']
        all_machines = set([file.machine for file in shot_files_all])
        train_machines = set([file.machine for file in shot_files])

        if train_machines >= all_machines:
            shot_files_use = shot_files
        else:
            print('Testing set contains new machine, using testing set to train normalizer for that machine.')
            shot_files_use = shot_files_all

        # shot_list_dir = conf['paths']['shot_list_dir']
        use_shots = max(400,conf['data']['use_shots'])
        return self.train_on_files(shot_files_use,use_shots)


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
            #for (i,stats) in enumerate(map(self.train_on_single_shot,shot_list_picked)):
                self.incorporate_stats(stats)
                self.machines.add(stats.machine)
                sys.stdout.write('\r' + '{}/{}'.format(i,len(shot_list_picked)))

            pool.close()
            pool.join()
            print('Finished Training Normalizer on {} files in {} seconds'.format(len(shot_list_picked),time.time()-start_time))
            self.save_stats()
        else:
            self.load_stats()
        print(self)


    def cut_end_of_shot(self,shot):
        cut_shot_ends = self.conf['data']['cut_shot_ends']
        if cut_shot_ends:
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
        #print(shot)
        stats = self.extract_stats(shot) 
        shot.make_light()
        return stats
    
    def ensure_save_directory(self):
        prepath = os.path.dirname(self.path)
        if not os.path.exists(prepath):
            os.makedirs(prepath)

    def previously_saved_stats(self):
        if not os.path.isfile(self.path):
            return False
        else:
            dat = np.load(self.path,encoding="latin1")
            machines = dat['machines'][()]
            ret =  all([m in machines for m in self.conf['paths']['all_machines']])
            if not ret:
                print(machines)
                print(self.conf['paths']['all_machines'])
                print('Not all machines present. Recomputing normalizer.')
            return ret

    # def get_indices_list(self):
    #     return get_signal_slices(self.conf['paths']['signals_dirs'])





class MeanVarNormalizer(Normalizer):
    def __init__(self,conf):
        Normalizer.__init__(self,conf)
        self.means = dict()
        self.stds = dict()

    def __str__(self):
        s = ''
        for machine in self.means:
                means = np.median(self.means[machine],axis=0)
                stds = np.median(self.stds[machine],axis=0)
                s += 'Machine: {}:\nMean Var Normalizer.\nmeans: {}\nstds: {}'.format(machine,means,stds)
        return s 

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
        stats.machine = shot.machine
        return stats



    def incorporate_stats(self,stats):
        machine = stats.machine
        self.ensure_machine(stats.machine)
        if stats.valid:
            means = stats.means
            stds = stats.stds
            if self.num_processed[machine] == 0:
                self.means[machine] = means
                self.stds[machine] = stds 
            else:
                self.means[machine] = np.concatenate((self.means[machine],means),axis=0)
                self.stds[machine] = np.concatenate((self.stds[machine],stds),axis=0)
            self.num_processed[machine] = self.num_processed[machine] + 1
            self.num_disruptive[machine] = self.num_disruptive[machine] + (1 if stats.is_disruptive else 0)


    def apply(self,shot):
        m = shot.machine
        assert self.means[m] is not None and self.stds[m] is not None, "self.means or self.stds not initialized"
        apply_positivity(shot)
        means = np.median(self.means[m],axis=0)
        stds = np.median(self.stds[m],axis=0)
        for (i,sig) in enumerate(shot.signals):
            if sig.normalize:
                stds_curr = stds[i]
                if stds_curr == 0.0:
                    stds_curr = 1.0
                shot.signals_dict[sig] = (shot.signals_dict[sig] - means[i])/stds_curr

        shot.ttd = self.remapper(shot.ttd,self.conf['data']['T_warning'])
        self.cut_end_of_shot(shot)
        # self.apply_positivity_mask(shot)
        # self.apply_mask(shot)


    def save_stats(self):
        # standard_deviations = dat['standard_deviations']
        # num_processed = dat['num_processed']
        # num_disruptive = dat['num_disruptive']
        self.ensure_save_directory()
        np.savez(self.path,means = self.means,stds = self.stds,
         num_processed=self.num_processed,num_disruptive=self.num_disruptive,machines=self.machines)
        print('saved normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))

    def load_stats(self):
        assert self.previously_saved_stats(), "stats not saved before"
        dat = np.load(self.path,encoding="latin1")
        self.means = dat['means'][()]
        self.stds = dat['stds'][()]
        self.num_processed = dat['num_processed'][()]
        self.num_disruptive = dat['num_disruptive'][()]
        self.machines = dat['machines'][()]
        for machine in self.means:
                print('Machine {}:'.format(machine))
                print('loaded normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))
        #print('loading normalization data from {} shots, {} disruptive'.format(num_processed,num_disruptive))


class VarNormalizer(MeanVarNormalizer):
    def apply(self,shot):
        assert self.means is not None and self.stds is not None, "self.means or self.stds not initialized"
        m = shot.machine
        apply_positivity(shot)
        stds = np.median(self.stds[m],axis=0)
        for (i,sig) in enumerate(shot.signals):
            if sig.normalize:
                stds_curr = stds[i]
                if stds_curr == 0.0:
                    stds_curr = 1.0
                shot.signals_dict[sig] = (shot.signals_dict[sig])/stds_curr
        shot.ttd = self.remapper(shot.ttd,self.conf['data']['T_warning'])
        self.cut_end_of_shot(shot)

    def __str__(self):
        s = ''
        for m in self.stds:
                stds = np.median(self.stds[m],axis=0)
                s += 'Machine: {}:\n'.format(m)
                s += 'Var Normalizer.\nstds: {}\n'.format(stds)
        return s


class AveragingVarNormalizer(VarNormalizer):

    def apply(self,shot):
        super(AveragingVarNormalizer,self).apply(shot)
        window_decay = self.conf['data']['window_decay']
        window_size = self.conf['data']['window_size']
        window = exponential(window_size,0,window_decay,False)
        window /= np.sum(window)
        apply_positivity(shot)
        for (i,sig) in enumerate(shot.signals):
            if sig.normalize:
                shot.signals_dict[sig] = apply_along_axis(lambda m : correlate(m,window,'valid'),axis=0,arr=shot.signals_dict[sig])
        shot.ttd = shot.ttd[-shot.signals.shape[0]:]

    def __str__(self):
        window_decay = self.conf['data']['window_decay']
        window_size = self.conf['data']['window_size']
        s = ''
        for m in self.stds:
                stds = np.median(self.stds[m],axis=0)
                s += 'Machine: {}:\n'.format(m)
                s += 'Averaging Var Normalizer.\nstds: {}\nWindow size: {}, Window decay: {}'.format(stds,window_size,window_decay)
        return s


class MinMaxNormalizer(Normalizer):
    def __init__(self,conf):
        Normalizer.__init__(self,conf)
        self.minimums = None
        self.maximums = None


    def __str__(self):
        s = ''
        for m in self.minimums:
                s += 'Machine {}:\n.Min Max Normalizer.\nminimums: {}\nmaximums: {}'.format(m,self.minimums[m],self.maximums[m])
        return s 

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
        stats.machine = shot.machine
        return stats


    def incorporate_stats(self,stats):
        self.ensure_machine(stats.machine)
        if stats.valid:
            m = stats.machine
            minimums = stats.minimums
            maximums = stats.maximums
            if self.num_processed == 0:
                self.minimums[m] = minimums
                self.maximums[m] = maximums
            else:
                self.minimums[m] = (self.num_processed[m]*self.minimums + minimums)/(self.num_processed[m] + 1.0)#snp.min(vstack((self.minimums,minimums)),0)
                self.maximums[m] = (self.num_processed[m]*self.maximums + maximums)/(self.num_processed[m] + 1.0)#snp.max(vstack((self.maximums,maximums)),0)
            self.num_processed[m] = self.num_processed[m] + 1
            self.num_disruptive[m] = self.num_disruptive[m] + (1 if stats.is_disruptive else 0)


    def apply(self,shot):
        assert(self.minimums is not None and self.maximums is not None) 
        m = shot.machine
        apply_positivity(shot)
        curr_range = (self.maximums[m] - self.minimums[m])
        if curr_range == 0.0:
                curr_range = 1.0
        shot.signals = (shot.signals - self.minimums[m])/curr_range
        for (i,sig) in enumerate(shot.signals):
            if sig.normalize:
                shot.signals_dict[sig] = (shot.signals_dict[sig] - self.minimums[m])/(self.maximums[m] - self.minimums[m])
        shot.ttd = self.remapper(shot.ttd,self.conf['data']['T_warning'])
        self.cut_end_of_shot(shot)
        # self.apply_positivity_mask(shot)
        # self.apply_mask(shot)

    def save_stats(self):
        # standard_deviations = dat['standard_deviations']
        # num_processed = dat['num_processed']
        # num_disruptive = dat['num_disruptive']
        self.ensure_save_directory()
        np.savez(self.path,minimums = self.minimums,maximums = self.maximums,
         num_processed=self.num_processed,num_disruptive=self.num_disruptive,machines=self.machines)
        print('saved normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))

    def load_stats(self):
        assert(self.previously_saved_stats())
        dat = np.load(self.path,encoding="latin1")
        self.minimums = dat['minimums'][()]
        self.maximums = dat['maximums'][()]
        self.num_processed = dat['num_processed'][()]
        self.num_disruptive = dat['num_disruptive'][()]
        self.machines = dat['machines'][()]
        print('loaded normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))
        #print('loading normalization data from {} shots, {} disruptive'.format(num_processed,num_disruptive))


def get_individual_shot_file(prepath,shot_num,ext='.txt'):
    return prepath + str(shot_num) + ext 

def apply_positivity(shot):
    for (i,sig) in enumerate(shot.signals):
        if sig.is_strictly_positive:
            print ('Applying positivity constraint to {} signal'.format(sig.description))
            shot.signals_dict[sig]=np.clip(shot.signals_dict[sig],0,np.inf)
