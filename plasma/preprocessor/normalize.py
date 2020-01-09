'''
#########################################################
This file containts classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
import plasma.global_vars as g
import os
import time
import sys
import abc

import numpy as np
from scipy.signal import exponential, correlate
import pathos.multiprocessing as mp

from plasma.primitives.shots import ShotList, Shot

'''TODO
- incorporate stats, pass machine (perhaps save machine in stats object!)
- incorporate stats, have a dictionary of aggregate stats for every machine.
- check "is_previously_saved" by making sure there is a normalizer for every
machine

'''


#################
# NORMALIZATION #
#################
class Stats(object):
    pass


class Normalizer(object):
    def __init__(self, conf):
        self.num_processed = dict()
        self.num_disruptive = dict()
        self.conf = conf
        self.path = conf['paths']['normalizer_path']
        self.remapper = conf['data']['target'].remapper
        self.machines = set()
        self.inference_mode = False
        self.bound = np.Inf
        if 'norm_stat_range' in self.conf['data']:
            self.bound = self.conf['data']['norm_stat_range']

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def extract_stats(self, shot):
        pass

    @abc.abstractmethod
    def incorporate_stats(self, stats):
        pass

    @abc.abstractmethod
    def apply(self, shot):
        pass

    @abc.abstractmethod
    def save_stats(self, verbose=False):
        pass

    @abc.abstractmethod
    def load_stats(self, verbose=False):
        pass

    def print_summary(self, action='loaded'):
        g.print_unique(
            '{} normalization data from {} shots ( {} disruptive )'.format(
                action, self.num_processed, self.num_disruptive))

    def set_inference_mode(self, val):
        self.inference_mode = val

    def ensure_machine(self, machine):
        if machine not in self.means:
            self.num_processed[machine] = 0
            self.num_disruptive[machine] = 0
    # Modify the above to change the specifics of the normalization scheme

    def train(self, verbose=False):
        conf = self.conf
        # only use training shots here!! "Don't touch testing shots"
        # + conf['paths']['shot_files_test']
        shot_files = conf['paths']['shot_files']
        shot_files_all = conf['paths']['shot_files_all']
        all_machines = set([file.machine for file in shot_files_all])
        train_machines = set([file.machine for file in shot_files])

        if train_machines >= all_machines:
            shot_files_use = shot_files
        else:
            print('Testing set contains new machine, using testing set ',
                  'to train normalizer for that machine.')
            shot_files_use = shot_files_all

        # shot_list_dir = conf['paths']['shot_list_dir']
        use_shots = max(400, conf['data']['use_shots'])
        return self.train_on_files(shot_files_use, use_shots, all_machines,
                                   verbose=verbose)

    def train_on_files(self, shot_files, use_shots, all_machines,
                       verbose=False):
        conf = self.conf
        all_signals = conf['paths']['all_signals']
        shot_list = ShotList()
        shot_list.load_from_shot_list_files_objects(shot_files, all_signals)
        shot_list_picked = shot_list.random_sublist(use_shots)

        previously_saved, machines_saved = self.previously_saved_stats()
        machines_to_compute = all_machines - machines_saved
        recompute = conf['data']['recompute_normalization']
        if recompute:
            machines_to_compute = all_machines
            previously_saved = False
        if not previously_saved or len(machines_to_compute) > 0:
            if previously_saved:
                self.load_stats(verbose=True)
            print('computing normalization for machines {}'.format(
                machines_to_compute))
            use_cores = max(1, mp.cpu_count()-2)
            pool = mp.Pool(use_cores)
            print('running in parallel on {} processes'.format(
                pool._processes))
            start_time = time.time()

            for (i, stats) in enumerate(pool.imap_unordered(
                    self.train_on_single_shot, shot_list_picked)):
                # for (i,stats) in
                # enumerate(map(self.train_on_single_shot,shot_list_picked)):
                if stats.machine in machines_to_compute:
                    self.incorporate_stats(stats)
                    self.machines.add(stats.machine)
                sys.stdout.write('\r'
                                 + '{}/{}'.format(i, len(shot_list_picked)))
            pool.close()
            pool.join()
            print('\nFinished Training Normalizer on ',
                  '{} files in {} seconds'.format(len(shot_list_picked),
                                                  time.time()-start_time))
            self.save_stats(verbose=True)
        else:
            self.load_stats(verbose=verbose)
        # print representation of trained Normalizer to stdout:
        # Machine, NormalizerName, per-signal normalization stats/params
        if verbose:
            g.print_unique(self)

    def cut_end_of_shot(self, shot):
        cut_shot_ends = self.conf['data']['cut_shot_ends']
        # only cut shots during training
        if not self.inference_mode and cut_shot_ends:
            T_min_warn = self.conf['data']['T_min_warn']
            if shot.ttd.shape[0] - T_min_warn <= max(
                    self.conf['model']['length'], 0):
                print("not cutting shot; length of shot after cutting by ",
                      "T_min_warn would be shorter than RNN length")
                return
            for key in shot.signals_dict:
                shot.signals_dict[key] = shot.signals_dict[key][:-T_min_warn,:]  # noqa
            shot.ttd = shot.ttd[:-T_min_warn]

    # def apply_mask(self,shot):
    #     use_signals = self.conf['paths']['use_signals']
    #     return shot.get_data_arrays(use_signals)

    # def apply_positivity_mask(self,shot):
    #     mask = self.conf['paths']['positivity_mask']
    #     mask = [np.array(subl) for subl in mask]
    #     indices = np.concatenate([indices_sublist[mask[i]] for
    #     i,indices_sublist in enumerate(self.get_indices_list())])
    #     shot.signals[:,indices] = np.clip(shot.signals[:,indices],0,np.Inf)

    def train_on_single_shot(self, shot):
        assert isinstance(shot, Shot), 'should be instance of shot'
        processed_prepath = self.conf['paths']['processed_prepath']
        shot.restore(processed_prepath)
        # print(shot)
        stats = self.extract_stats(shot)
        shot.make_light()
        return stats

    def ensure_save_directory(self):
        prepath = os.path.dirname(self.path)
        if not os.path.exists(prepath):
            os.makedirs(prepath)

    def previously_saved_stats(self):
        if not os.path.isfile(self.path):
            return False, set([])
        else:
            dat = np.load(self.path, encoding="latin1", allow_pickle=True)
            machines = dat['machines'][()]
            ret = all(
                [m in machines for m in self.conf['paths']['all_machines']])
            if not ret:
                print(machines)
                print(self.conf['paths']['all_machines'])
                print('Not all machines present. Recomputing normalizer.')
            return True, set(machines)

    # def get_indices_list(self):
    #     return get_signal_slices(self.conf['paths']['signals_dirs'])


class MeanVarNormalizer(Normalizer):
    def __init__(self, conf):
        Normalizer.__init__(self, conf)
        self.means = dict()
        self.stds = dict()
        self.bound = np.Inf
        if 'norm_stat_range' in self.conf['data']:
            self.bound = self.conf['data']['norm_stat_range']

    def __str__(self):
        s = ''
        for machine in self.means:
            means = np.median(self.means[machine], axis=0)
            stds = np.median(self.stds[machine], axis=0)
            s += 'Machine = {}:\nMean Var Normalizer.\n'.format(machine)
            s += 'means: {}\nstds: {}'.format(means, stds)
        return s

    def extract_stats(self, shot):
        stats = Stats()
        if shot.valid:
            list_of_signals = shot.get_individual_signal_arrays()
            num_signals = len(list_of_signals)
            stats.means = np.reshape(np.array([np.mean(sig) for
                                               sig in list_of_signals]),
                                     (1, num_signals))
            stats.stds = np.reshape(np.array([np.std(sig, dtype=np.float64) for
                                              sig in list_of_signals]),
                                    (1, num_signals))
            stats.is_disruptive = shot.is_disruptive
        else:
            print('Warning: shot {} not valid [omit]'.format(shot.number))
        stats.valid = shot.valid
        stats.machine = shot.machine
        return stats

    def incorporate_stats(self, stats):
        machine = stats.machine
        self.ensure_machine(stats.machine)
        if stats.valid:
            means = stats.means
            stds = stats.stds
            if self.num_processed[machine] == 0:
                self.means[machine] = means
                self.stds[machine] = stds
            else:
                self.means[machine] = np.concatenate(
                    (self.means[machine], means), axis=0)
                self.stds[machine] = np.concatenate(
                    (self.stds[machine], stds), axis=0)
            self.num_processed[machine] = self.num_processed[machine] + 1
            self.num_disruptive[machine] = (
                self.num_disruptive[machine]
                + (1 if stats.is_disruptive else 0))

    def apply(self, shot):
        apply_positivity(shot)
        m = shot.machine
        assert self.means[m] is not None and self.stds[m] is not None, (
            "self.means or self.stds not initialized")
        means = np.median(self.means[m], axis=0)
        stds = np.median(self.stds[m], axis=0)
        for (i, sig) in enumerate(shot.signals):
            if sig.normalize:
                stds_curr = stds[i]
                if stds_curr == 0.0:
                    stds_curr = 1.0
                shot.signals_dict[sig] = (
                    shot.signals_dict[sig] - means[i])/stds_curr
                shot.signals_dict[sig] = np.clip(
                    shot.signals_dict[sig], -self.bound, self.bound)

        shot.ttd = self.remapper(shot.ttd, self.conf['data']['T_warning'])
        self.cut_end_of_shot(shot)
        # self.apply_positivity_mask(shot)
        # self.apply_mask(shot)

    def save_stats(self, verbose=False):
        # standard_deviations = dat['standard_deviations']
        # num_processed = dat['num_processed']
        # num_disruptive = dat['num_disruptive']
        self.ensure_save_directory()
        np.savez(self.path, means=self.means, stds=self.stds,
                 num_processed=self.num_processed,
                 num_disruptive=self.num_disruptive, machines=self.machines)
        if verbose:
            self.print_summary(action='saved')

    def load_stats(self, verbose=False):
        assert self.previously_saved_stats()[0], "stats not saved before"
        dat = np.load(self.path, encoding="latin1", allow_pickle=True)
        self.means = dat['means'][()]
        self.stds = dat['stds'][()]
        self.num_processed = dat['num_processed'][()]
        self.num_disruptive = dat['num_disruptive'][()]
        self.machines = dat['machines'][()]
        # for machine in self.means:
        #     g.print_unique('Machine = {}:'.format(machine))
        if verbose:
            self.print_summary()


class VarNormalizer(MeanVarNormalizer):
    def apply(self, shot):
        apply_positivity(shot)
        assert self.means is not None and self.stds is not None, (
            "self.means or self.stds not initialized")
        m = shot.machine
        stds = np.median(self.stds[m], axis=0)
        for (i, sig) in enumerate(shot.signals):
            if sig.normalize:
                stds_curr = stds[i]
                if stds_curr == 0.0:
                    stds_curr = 1.0
                shot.signals_dict[sig] = (shot.signals_dict[sig])/stds_curr
                shot.signals_dict[sig] = np.clip(
                    shot.signals_dict[sig], -self.bound, self.bound)
        shot.ttd = self.remapper(shot.ttd, self.conf['data']['T_warning'])
        self.cut_end_of_shot(shot)

    def __str__(self):
        s = ''
        for m in self.stds:
            stds = np.median(self.stds[m], axis=0)
            s += 'Machine: {}:\n'.format(m)
            s += 'Var Normalizer.\nstds: {}\n'.format(stds)
        return s


class AveragingVarNormalizer(VarNormalizer):
    def apply(self, shot):
        apply_positivity(shot)
        super(AveragingVarNormalizer, self).apply(shot)
        window_decay = self.conf['data']['window_decay']
        window_size = self.conf['data']['window_size']
        window = exponential(window_size, 0, window_decay, False)
        window /= np.sum(window)
        for (i, sig) in enumerate(shot.signals):
            if sig.normalize:
                shot.signals_dict[sig] = np.apply_along_axis(
                    lambda m: correlate(m, window, 'valid'),
                    axis=0, arr=shot.signals_dict[sig])
                shot.signals_dict[sig] = np.clip(
                    shot.signals_dict[sig], -self.bound, self.bound)
        shot.ttd = shot.ttd[-shot.signals.shape[0]:]

    def __str__(self):
        window_decay = self.conf['data']['window_decay']
        window_size = self.conf['data']['window_size']
        s = ''
        for m in self.stds:
            stds = np.median(self.stds[m], axis=0)
            s += 'Machine: {}:\n'.format(m)
            s += 'Averaging Var Normalizer.\nstds: '
            s += ' {}\nWindow size: {}, Window decay: {}'.format(
                stds, window_size, window_decay)
        return s


class MinMaxNormalizer(Normalizer):
    def __init__(self, conf):
        Normalizer.__init__(self, conf)
        self.minimums = None
        self.maximums = None
        self.bound = np.Inf
        if 'norm_stat_range' in self.conf['data']:
            self.bound = self.conf['data']['norm_stat_range']

    def __str__(self):
        s = ''
        for m in self.minimums:
            s += 'Machine {}:\n.Min Max Normalizer.\n'.format(m,
                                                              self.minimums[m])
            s += 'minimums: {}\nmaximums: {}'.format(self.maximums[m])
        return s

    def extract_stats(self, shot):
        stats = Stats()
        if shot.valid:
            list_of_signals = shot.get_individual_signal_arrays()
            stats.minimums = np.array([np.min(sig) for sig in list_of_signals])
            stats.maximums = np.array([np.max(sig) for sig in list_of_signals])
            stats.is_disruptive = shot.is_disruptive
        else:
            print('Warning: shot {} not valid [omit]'.format(shot.number))
        stats.valid = shot.valid
        stats.machine = shot.machine
        return stats

    def incorporate_stats(self, stats):
        self.ensure_machine(stats.machine)
        if stats.valid:
            m = stats.machine
            minimums = stats.minimums
            maximums = stats.maximums
            if self.num_processed == 0:
                self.minimums[m] = minimums
                self.maximums[m] = maximums
            else:
                self.minimums[m] = (self.num_processed[m]*self.minimums
                                    + minimums)/(self.num_processed[m] + 1.0)
                self.maximums[m] = (self.num_processed[m]*self.maximums
                                    + maximums)/(self.num_processed[m] + 1.0)
            self.num_processed[m] = self.num_processed[m] + 1
            self.num_disruptive[m] = (self.num_disruptive[m]
                                      + (1 if stats.is_disruptive else 0))

    def apply(self, shot):
        apply_positivity(shot)
        assert self.minimums is not None and self.maximums is not None
        m = shot.machine
        curr_range = (self.maximums[m] - self.minimums[m])
        if curr_range == 0.0:
            curr_range = 1.0
        shot.signals = (shot.signals - self.minimums[m])/curr_range
        for (i, sig) in enumerate(shot.signals):
            if sig.normalize:
                shot.signals_dict[sig] = (
                    shot.signals_dict[sig] - self.minimums[m])/(
                        self.maximums[m] - self.minimums[m])
                shot.signals_dict[sig] = np.clip(
                    shot.signals_dict[sig], -self.bound, self.bound)
        shot.ttd = self.remapper(shot.ttd, self.conf['data']['T_warning'])
        self.cut_end_of_shot(shot)
        # self.apply_positivity_mask(shot)
        # self.apply_mask(shot)

    def save_stats(self, verbose=False):
        # standard_deviations = dat['standard_deviations']
        # num_processed = dat['num_processed']
        # num_disruptive = dat['num_disruptive']
        self.ensure_save_directory()
        np.savez(self.path, minimums=self.minimums, maximums=self.maximums,
                 num_processed=self.num_processed,
                 num_disruptive=self.num_disruptive, machines=self.machines)
        if verbose:
            self.print_summary(action='saved')

    def load_stats(self, verbose=False):
        assert self.previously_saved_stats()[0]
        dat = np.load(self.path, encoding="latin1", allow_pickle=True)
        self.minimums = dat['minimums'][()]
        self.maximums = dat['maximums'][()]
        self.num_processed = dat['num_processed'][()]
        self.num_disruptive = dat['num_disruptive'][()]
        self.machines = dat['machines'][()]
        # for machine in self.means:
        #     g.print_unique('Machine {}:'.format(machine))
        if verbose:
            self.print_summary()


def apply_positivity(shot):
    for (i, sig) in enumerate(shot.signals):
        if hasattr(sig, "is_strictly_positive"):
            # backwards compatibility when this attribute didn't exist
            if sig.is_strictly_positive:
                # print ('Applying positivity constraint to {}
                # signal'.format(sig.description))
                shot.signals_dict[sig] = np.clip(
                    shot.signals_dict[sig], 0, np.inf)
