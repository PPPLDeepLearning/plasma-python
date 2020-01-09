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

from plasma.utils.processing import (
    train_test_split, cut_and_resample_signal,
    get_individual_shot_file
    )
from plasma.utils.downloading import makedirs_process_safe
from plasma.utils.hashing import myhash


class ShotListFiles(object):
    def __init__(self, machine, prepath, paths, description=''):
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

    def get_single_shot_numbers_and_disruption_times(self, full_path):
        data = np.loadtxt(full_path, ndmin=1, dtype={
            'names': ('num', 'disrupt_times'), 'formats': ('i4', 'f4')})
        shots = np.array(list(zip(*data))[0])
        disrupt_times = np.array(list(zip(*data))[1])
        return shots, disrupt_times

    def get_shot_numbers_and_disruption_times(self):
        all_shots = []
        all_disruption_times = []
        # all_machines_arr = []
        for path in self.paths:
            full_path = self.prepath + path
            shots, disruption_times = (
                self.get_single_shot_numbers_and_disruption_times(full_path))
            all_shots.append(shots)
            all_disruption_times.append(disruption_times)
        return np.concatenate(all_shots), np.concatenate(all_disruption_times)


class ShotList(object):
    '''
    A wrapper class around list of Shot objects, providing utilities to
    extract, load and transform Shots before passing them to an estimator.

    During distributed training, shot lists are split into sublists.
    A sublist is a ShotList object having num_at_once shots. The ShotList
    contains an entire dataset as specified in the configuration file.
    '''

    def __init__(self, shots=None):
        '''
        A ShotList is a list of 2D Numpy arrays.
        '''
        self.shots = []
        if shots is not None:
            assert all([isinstance(shot, Shot) for shot in shots])
            self.shots = [shot for shot in shots]

    def load_from_shot_list_files_object(self, shot_list_files_object,
                                         signals):
        machine = shot_list_files_object.machine
        shot_numbers, disruption_times = (
            shot_list_files_object.get_shot_numbers_and_disruption_times())
        for number, t in list(zip(shot_numbers, disruption_times)):
            self.append(Shot(number=number, t_disrupt=t, machine=machine,
                             signals=[s for s in signals if
                                      s.is_defined_on_machine(machine)]))

    def load_from_shot_list_files_objects(self, shot_list_files_objects,
                                          signals):
        for obj in shot_list_files_objects:
            self.load_from_shot_list_files_object(obj, signals)

    def split_train_test(self, conf):
        # shot_list_dir = conf['paths']['shot_list_dir']
        shot_files = conf['paths']['shot_files']
        shot_files_test = conf['paths']['shot_files_test']
        train_frac = conf['training']['train_frac']
        shuffle_training = conf['training']['shuffle_training']
        use_shots = conf['data']['use_shots']
        all_signals = conf['paths']['all_signals']
        # split "maximum number of shots to use" into:
        # test vs. (train U validate)
        use_shots_train = int(round(train_frac*use_shots))
        use_shots_test = int(round((1-train_frac)*use_shots))
        if len(shot_files_test) == 0:
            # split randomly, e.g. sample both sets from same distribution
            # such as D3D test and train
            shot_list_train, shot_list_test = train_test_split(
                self.shots, train_frac, shuffle_training)
        # train and test list given, e.g. they are sampled from separate
        # distributions such as train=CW and test=ILW for JET
        else:
            shot_list_train = ShotList()
            shot_list_train.load_from_shot_list_files_objects(
                shot_files, all_signals)

            shot_list_test = ShotList()
            shot_list_test.load_from_shot_list_files_objects(
                shot_files_test, all_signals)

        shot_numbers_train = [shot.number for shot in shot_list_train]
        shot_numbers_test = [shot.number for shot in shot_list_test]
        # make sure we only use pre-filtered valid shots
        shots_train = self.filter_by_number(shot_numbers_train)
        shots_test = self.filter_by_number(shot_numbers_test)
        return shots_train.random_sublist(
            use_shots_train), shots_test.random_sublist(use_shots_test)

    def split_direct(self, frac, do_shuffle=True):
        shot_list_one, shot_list_two = train_test_split(
            self.shots, frac, do_shuffle)
        return ShotList(shot_list_one), ShotList(shot_list_two)

    def filter_by_number(self, numbers):
        new_shot_list = ShotList()
        numbers = set(numbers)
        for shot in self.shots:
            if shot.number in numbers:
                new_shot_list.append(shot)
        return new_shot_list

    def set_weights(self, weights):
        assert len(weights) == len(self.shots)
        for (i, w) in enumerate(weights):
            self.shots[i].weight = w

    def sample_weighted_given_arr(self, p):
        p = p/np.sum(p)
        idx = np.random.choice(range(len(self.shots)), p=p)
        return self.shots[idx]

    def sample_shot(self):
        idx = np.random.choice(range(len(self.shots)))
        return self.shots[idx]

    def sample_weighted(self):
        p = np.array([shot.weight for shot in self.shots])
        return self.sample_weighted_given_arr(p)

    def sample_single_class(self, disruptive):
        weights_d = 0.0
        weights_nd = 1.0
        if disruptive:
            weights_d = 1.0
            weights_nd = 0.0
        p = np.array([weights_d if shot.is_disruptive_shot()
                      else weights_nd for shot in self.shots])
        return self.sample_weighted_given_arr(p)

    def sample_equal_classes(self):
        weights_d, weights_nd = self.get_weights_d_nd()
        p = np.array([weights_d if shot.is_disruptive_shot()
                      else weights_nd for shot in self.shots])
        return self.sample_weighted_given_arr(p)

    def get_weights_d_nd(self):
        # TODO(KGF): only called in above sample_equal_classes()
        num_total = len(self)
        num_d = self.num_disruptive()
        num_nd = num_total - num_d
        if num_nd == 0 or num_d == 0:
            weights_d = 1.0
            weights_nd = 1.0
        else:
            weights_d = 1.0*num_nd
            weights_nd = 1.0*num_d
        max_weight = np.maximum(weights_d, weights_nd)
        return weights_d/max_weight, weights_nd/max_weight

    def num_timesteps(self, prepath):
        ls = [shot.num_timesteps(prepath) for shot in self.shots]
        timesteps_total = sum(ls)
        timesteps_d = sum([ts for (i, ts) in enumerate(
            ls) if self.shots[i].is_disruptive_shot()])
        timesteps_nd = timesteps_total-timesteps_d
        return timesteps_total, timesteps_d, timesteps_nd

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

    def __add__(self, other_list):
        return ShotList(self.shots + other_list.shots)

    def index(self, item):
        return self.shots.index(item)

    def __getitem__(self, key):
        return self.shots[key]

    def random_sublist(self, num):
        num = min(num, len(self))
        shots_picked = np.random.choice(self.shots, size=num, replace=False)
        return ShotList(shots_picked)

    def sublists(self, num, do_shuffle=True, equal_size=False):
        lists = []
        if do_shuffle:
            self.shuffle()
        for i in range(0, len(self), num):
            subl = self.shots[i:i+num]
            while equal_size and len(subl) < num:
                subl.append(rnd.choice(self.shots))
            lists.append(subl)
        return [ShotList(l) for l in lists]

    def shuffle(self):
        np.random.shuffle(self.shots)

    def sort(self):
        self.shots.sort()  # will sort based on machine and number

    def as_list(self):
        return self.shots

    def append(self, shot):
        assert isinstance(shot, Shot)
        self.shots.append(shot)

    def remove(self, shot):
        assert shot in self.shots
        self.shots.remove(shot)
        assert shot not in self.shots

    def make_light(self):
        for shot in self.shots:
            shot.make_light()

    def append_if_valid(self, shot):
        if shot.valid:
            self.append(shot)
            return True
        else:
            # print('Warning: shot {} not valid [omit]'.format(shot.number))
            return False


class Shot(object):
    '''
    A class representing a shot.
    Each shot is a measurement of plasma properties (current, locked mode
    amplitude, etc.) as a function of time.

    For 0D data, each shot is modeled as a 2D Numpy array - time vs a plasma
    property.
    '''

    def __init__(self, number=None, machine=None, signals=None,
                 signals_dict=None, ttd=None, valid=None, is_disruptive=None,
                 t_disrupt=None):
        '''
        Shot objects contain following attributes:

         - number: integer, unique identifier of a shot
         - t_disrupt: double, disruption time in milliseconds (second column in
         the shotlist input file)

         - ttd: Numpy array of doubles, time profile of the shot converted to
           time-to-disruption values
         - valid: boolean flag indicating whether plasma property
           (specifically, current) reaches a certain value during the shot
         - is_disruptive: boolean flag indicating whether a shot is disruptive
        '''
        self.number = number  # Shot number
        self.machine = machine  # machine on which it is defined
        self.signals = signals
        self.signals_dict = signals_dict
        self.ttd = ttd
        self.valid = valid
        self.is_disruptive = is_disruptive
        self.t_disrupt = t_disrupt
        self.weight = 1.0
        self.augmentation_fn = None
        if t_disrupt is not None:
            self.is_disruptive = Shot.is_disruptive_given_disruption_time(
                t_disrupt)
        else:
            print('Warning, disruption time (disruptivity) not set! ',
                  'Either set t_disrupt or is_disruptive')

    def get_id_str(self):
        return '{} : {}'.format(self.machine, self.number)

    def __lt__(self, other):
        return self.get_id_str().__lt__(other.get_id_str())

    def __eq__(self, other):
        return self.get_id_str().__eq__(other.get_id_str())

    def __hash__(self):
        return myhash(self.get_id_str())

    def __str__(self):
        string = 'number: {}\n'.format(self.number)
        string += 'machine: {}\n'.format(self.machine)
        string += 'signals: {}\n'.format(self.signals)
        string += 'signals_dict: {}\n'.format(self.signals_dict)
        string += 'ttd: {}\n'.format(self.ttd)
        string += 'valid: {}\n'.format(self.valid)
        string += 'is_disruptive: {}\n'.format(self.is_disruptive)
        string += 't_disrupt: {}\n'.format(self.t_disrupt)
        return string

    def num_timesteps(self, prepath):
        self.restore(prepath)
        ts = self.ttd.shape[0]
        self.make_light()
        return ts

    def get_number(self):
        return self.number

    def get_signals(self):
        return self.signals

    def is_valid(self):
        return self.valid

    def is_disruptive_shot(self):
        return self.is_disruptive

    def get_data_arrays(self, use_signals, dtype='float32'):
        t_array = self.ttd
        signal_array = np.zeros(
            (len(t_array), sum([sig.num_channels for sig in use_signals])),
            dtype=dtype)
        curr_idx = 0
        for sig in use_signals:
            signal_array[:, curr_idx:curr_idx
                         + sig.num_channels] = self.signals_dict[sig]
            curr_idx += sig.num_channels
        return t_array, signal_array

    def get_individual_signal_arrays(self):
        # guarantee ordering
        return [self.signals_dict[sig] for sig in self.signals]

    def preprocess(self, conf):
        sys.stdout.write('\rrecomputing {}'.format(self.number))
        sys.stdout.flush()
        # get minmax times
        time_arrays, signal_arrays, t_min, t_max, valid = (
            self.get_signals_and_times_from_file(conf))
        self.valid = valid
        # cut and resample
        if self.valid:
            self.cut_and_resample_signals(
                time_arrays, signal_arrays, t_min, t_max, conf)

    def get_signals_and_times_from_file(self, conf):
        valid = True
        t_min = -np.Inf
        t_max = np.Inf
        # t_thresh = -1
        signal_arrays = []
        time_arrays = []
        garbage = False
        # disruptive = self.t_disrupt >= 0
        # TODO(KGF): refactor the dataset-specific shot filtering settings
        # (expose in conf.yaml?)
        if conf['paths']['data'] == 'd3d_data_garbage':
            garbage = True
        invalid_signals = 0
        signal_prepath = conf['paths']['signal_prepath']
        # TODO(KGF): check the purpose of the following D3D-specific lines
        # added from fork in Dec 2019. Add [omit] print?
        # --- possibly corrupted raw shot files from original D3D shot set
        # if self.number in [127613, 129423, 125726, 126662]:
        #     return None, None, None, None, False
        for (i, signal) in enumerate(self.signals):
            if isinstance(signal_prepath, list):
                for prepath in signal_prepath:
                    t, sig, valid_signal = signal.load_data(
                        prepath, self, conf['data']['floatx'])
                if valid_signal:
                    break
            else:
                t, sig, valid_signal = signal.load_data(
                    signal_prepath, self, conf['data']['floatx'])
            if not valid_signal:
                # TODO(KGF): new check added from fork in Dec 2019.
                # Add [omit] print?
                if (signal.is_ip or 'q95' in signal.description
                        or garbage is False or sig is None):
                    # Exclude entire shot if missing plasma current or q95
                    return None, None, None, None, False
                else:
                    t = np.arange(0, 20, 0.001)
                    sig = np.zeros((t.shape[0], sig[1]))
                    invalid_signals += 1
                    signal_arrays.append(sig)
                    time_arrays.append(t)
            else:
                assert len(sig.shape) == 2
                assert len(t.shape) == 1
                assert len(t) > 1
                t_min = max(t_min, np.min(t))
                signal_arrays.append(sig)
                time_arrays.append(t)

                if self.is_disruptive and self.t_disrupt > np.max(t):
                    tol = signal.get_data_avail_tolerance(self.machine)
                    t_max_total = np.max(t) + tol
                    if self.t_disrupt > t_max_total:
                        if garbage is False:
                            print('Shot {}: disruption event '.format(
                                self.number),
                                  'is not contained in valid time region of ',
                                  'signal {} by {}s [omit]'.format(
                                      self.number, signal,
                                      self.t_disrupt - np.max(t)))
                            valid = False
                        else:
                            # Set the entire channel to zero to prevent any
                            # peeking into possible disruptions from this early
                            # terminated channel
                            invalid_signals += 1
                            t = np.arange(0, 20, 0.001)
                            sig = np.zeros((t.shape[0], sig.shape[1]))
                    else:
                        t_max = np.max(t) + tol
                else:
                    t_max = min(t_max, np.max(t))

        # make sure the shot is long enough.
        dt = conf['data']['dt']
        if (t_max - t_min)/dt <= (2*conf['model']['length']
                                  + conf['data']['T_min_warn']):
            print('Shot {} contains insufficient data [omit]'.format(
                self.number))
            valid = False

        assert t_max > t_min or not valid, (
            "t max: {}, t_min: {}".format(t_max, t_min))

        if self.is_disruptive:
            assert self.t_disrupt <= t_max or not valid
            t_max = self.t_disrupt
        if invalid_signals > 3:
            # Omit shot if more than 3 channels are bad
            print('Shot {}: has more than 3 invalid channels [omit]'.format(
                self.number))
            valid = False
        # if the signal has np.max(t) < t_disrupt, but t_max_total (with
        # positive tolerance) > t_disrupt, then the signal is implicitly
        # padded with 0's
        return time_arrays, signal_arrays, t_min, t_max, valid

    def cut_and_resample_signals(self, time_arrays, signal_arrays, t_min,
                                 t_max, conf):
        dt = conf['data']['dt']
        signals_dict = dict()

        # resample signals
        assert len(signal_arrays) == len(time_arrays) == len(self.signals)
        assert len(signal_arrays) > 0
        tr = 0
        for (i, signal) in enumerate(self.signals):
            tr, sigr = cut_and_resample_signal(
                time_arrays[i], signal_arrays[i], t_min, t_max, dt,
                conf['data']['floatx'])
            signals_dict[signal] = sigr

        ttd = self.convert_to_ttd(tr, conf)
        self.signals_dict = signals_dict
        self.ttd = ttd

    def convert_to_ttd(self, tr, conf):
        T_max = conf['data']['T_max']
        dt = conf['data']['dt']
        if self.is_disruptive:
            ttd = max(tr) - tr
            ttd = np.clip(ttd, 0, T_max)
        else:
            ttd = T_max*np.ones_like(tr)
        ttd = np.log10(ttd + 1.0*dt/10)
        return ttd

    def save(self, prepath):
        makedirs_process_safe(prepath)
        save_path = self.get_save_path(prepath)
        np.savez(save_path, valid=self.valid, is_disruptive=self.is_disruptive,
                 signals_dict=self.signals_dict, ttd=self.ttd)
        print('...saved shot {}'.format(self.number))

    def get_save_path(self, prepath):
        return get_individual_shot_file(prepath, self.machine, self.number,
                                        ext='.npz')

    def restore(self, prepath, light=False):
        assert self.previously_saved(prepath), 'shot was never saved'
        save_path = self.get_save_path(prepath)
        dat = np.load(save_path, encoding="latin1", allow_pickle=True)

        self.valid = dat['valid'][()]
        self.is_disruptive = dat['is_disruptive'][()]

        if light:
            self.signals_dict = None
            self.ttd = None
        else:
            self.signals_dict = dat['signals_dict'][()]
            self.ttd = dat['ttd']

    def previously_saved(self, prepath):
        save_path = self.get_save_path(prepath)
        return os.path.isfile(save_path)

    def make_light(self):
        self.signals_dict = None
        self.ttd = None

    @staticmethod
    def is_disruptive_given_disruption_time(t):
        return t >= 0
