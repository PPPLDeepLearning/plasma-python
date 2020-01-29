'''
#########################################################
This file contains classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
import plasma.global_vars as g
from os import listdir  # , remove
import time
import sys
import os

import numpy as np
import pathos.multiprocessing as mp

from plasma.utils.processing import append_to_filename
from plasma.utils.diagnostics import print_shot_list_sizes
from plasma.primitives.shots import ShotList
from plasma.utils.downloading import mkdirdepth


class Preprocessor(object):
    def __init__(self, conf):
        self.conf = conf

    def clean_shot_lists(self):
        shot_list_dir = self.conf['paths']['shot_list_dir']
        paths = [os.path.join(shot_list_dir, f) for f in
                 listdir(shot_list_dir) if
                 os.path.isfile(os.path.join(shot_list_dir, f))]
        for path in paths:
            self.clean_shot_list(path)

    def clean_shot_list(self, path):
        data = np.loadtxt(path)
        # ending_idx = path.rfind('.')
        new_path = append_to_filename(path, '_clear')
        if len(np.shape(data)) < 2:
            # nondisruptive
            nd_times = -1.0*np.ones_like(data)
            data_two_column = np.vstack((data, nd_times)).transpose()
            np.savetxt(new_path, data_two_column, fmt='%d %f')
            print('created new file: {}'.format(new_path))
            print('deleting old file: {}'.format(path))
            os.remove(path)

    def all_are_preprocessed(self):
        return os.path.isfile(self.get_shot_list_path())

    def preprocess_all(self):
        conf = self.conf
        shot_files_all = conf['paths']['shot_files_all']
        # shot_files_train = conf['paths']['shot_files']
        # shot_files_test = conf['paths']['shot_files_test']
        # shot_list_dir = conf['paths']['shot_list_dir']
        use_shots = conf['data']['use_shots']
        # train_frac = conf['training']['train_frac']
        # use_shots_train = int(round(train_frac*use_shots))
        # use_shots_test = int(round((1-train_frac)*use_shots))
        #        print(use_shots_train)
        #        print(use_shots_test) #each print out 100,000

        # if len(shot_files_test) > 0:
        #     return
        #     self.preprocess_from_files(shot_list_dir,shot_files_train,
        #      machines_train,use_shots_train)
        #     + self.preprocess_from_files(shot_list_dir,
        #      shot_files_test,machines_train,use_shots_test)
        # else:
        return self.preprocess_from_files(shot_files_all, use_shots)

    def preprocess_from_files(self, shot_files, use_shots):
        # all shots, including invalid ones
        all_signals = self.conf['paths']['all_signals']
        shot_list = ShotList()
        shot_list.load_from_shot_list_files_objects(shot_files, all_signals)
        shot_list_picked = shot_list.random_sublist(use_shots)

        # empty
        used_shots = ShotList()

        # TODO(KGF): generalize the follwowing line to perform well on
        # architecutres other than CPUs, e.g. KNLs
        # min( <desired-maximum-process-count>, max(1,mp.cpu_count()-2) )
        use_cores = max(1, mp.cpu_count() - 2)
        pool = mp.Pool(use_cores)
        print('Running in parallel on {} processes'.format(pool._processes))
        start_time = time.time()
        for (i, shot) in enumerate(pool.imap_unordered(
                self.preprocess_single_file, shot_list_picked)):
            # for (i,shot) in
            # enumerate(map(self.preprocess_single_file,shot_list_picked)):
            sys.stdout.write('\r{}/{}'.format(i, len(shot_list_picked)))
            used_shots.append_if_valid(shot)

        pool.close()
        pool.join()
        print('\nFinished preprocessing {} files in {} seconds'.format(
            len(shot_list_picked), time.time() - start_time))
        print('Using {} shots ({} disruptive shots)'.format(
            len(used_shots), used_shots.num_disruptive()))
        print('Omitted {} shots of {} total shots'.format(
            len(shot_list_picked) - len(used_shots), len(shot_list_picked)))
        print(
            'Omitted {} disruptive shots of {} total disruptive shots'.format(
                shot_list_picked.num_disruptive()
                - used_shots.num_disruptive(),
                shot_list_picked.num_disruptive()))

        if len(used_shots) == 0:
            print("WARNING: All shots were omitted, please ensure raw data "
                  " is complete and available at {}.".format(
                      self.conf['paths']['signal_prepath']))
        return used_shots

    def preprocess_single_file(self, shot):
        processed_prepath = self.conf['paths']['processed_prepath']
        recompute = self.conf['data']['recompute']
        # print('({}/{}): '.format(num_processed,use_shots))
        if recompute or not shot.previously_saved(processed_prepath):
            shot.preprocess(self.conf)
            shot.save(processed_prepath)
        else:
            try:
                shot.restore(processed_prepath, light=True)
                sys.stdout.write('\r{} exists.'.format(shot.number))
            except BaseException:
                shot.preprocess(self.conf)
                shot.save(processed_prepath)
                sys.stdout.write('\r{} exists but corrupted, resaved.'.format(
                    shot.number))
        shot.make_light()
        return shot

    def get_individual_channel_dirs(self):
        # TODO(KGF): unused
        return self.conf['paths']['signals_dirs']

    def get_shot_list_path(self):
        return self.conf['paths']['saved_shotlist_path']

    def load_shotlists(self):
        path = self.get_shot_list_path()
        data = np.load(path, encoding="latin1", allow_pickle=True)
        shot_list_train = data['shot_list_train'][()]
        shot_list_validate = data['shot_list_validate'][()]
        shot_list_test = data['shot_list_test'][()]
        if isinstance(shot_list_train, ShotList):
            return shot_list_train, shot_list_validate, shot_list_test
        else:
            return ShotList(shot_list_train), ShotList(
                shot_list_validate), ShotList(shot_list_test)

    def save_shotlists(self, shot_list_train, shot_list_validate,
                       shot_list_test):
        path = self.get_shot_list_path()
        mkdirdepth(path)
        np.savez(path, shot_list_train=shot_list_train,
                 shot_list_validate=shot_list_validate,
                 shot_list_test=shot_list_test)


def apply_bleed_in(conf, shot_list_train, shot_list_validate, shot_list_test):
    np.random.seed(2)
    num = conf['data']['bleed_in']
    # new_shots = []
    if num > 0:
        shot_list_bleed = ShotList()
        print('applying bleed in with {} disruptive shots\n'.format(num))
        # num_total = len(shot_list_test)
        num_d = shot_list_test.num_disruptive()
        # num_nd = num_total - num_d
        assert num_d >= num, (
            "Not enough disruptive shots {} to cover bleed in {}".format(
                num_d, num))
        num_sampled_d = 0
        num_sampled_nd = 0
        while num_sampled_d < num:
            s = shot_list_test.sample_shot()
            shot_list_bleed.append(s)
            if conf['data']['bleed_in_remove_from_test']:
                shot_list_test.remove(s)
            if s.is_disruptive:
                num_sampled_d += 1
            else:
                num_sampled_nd += 1
        print("Sampled {} shots, {} disruptive, {} nondisruptive".format(
            num_sampled_nd+num_sampled_d, num_sampled_d, num_sampled_nd))
        print("Before adding: training shots: {} validation shots: {}".format(
            len(shot_list_train), len(shot_list_validate)))
        assert num_sampled_d == num
        # add bleed-in shots to training and validation set repeatedly
        if conf['data']['bleed_in_equalize_sets']:
            print("Applying equalized bleed in")
            for shot_list_curr in [shot_list_train, shot_list_validate]:
                for i in range(len(shot_list_curr)):
                    s = shot_list_bleed.sample_shot()
                    shot_list_curr.append(s)
        elif conf['data']['bleed_in_repeat_fac'] > 1:
            repeat_fac = conf['data']['bleed_in_repeat_fac']
            print("Applying bleed in with repeat factor {}".format(repeat_fac))
            num_to_sample = int(round(repeat_fac*len(shot_list_bleed)))
            for i in range(num_to_sample):
                s = shot_list_bleed.sample_shot()
                shot_list_train.append(s)
                shot_list_validate.append(s)
        else:  # add each shot only once
            print("Applying bleed in without repetition")
            for s in shot_list_bleed:
                shot_list_train.append(s)
                shot_list_validate.append(s)
        print("After adding: training shots: {} validation shots: {}".format(
            len(shot_list_train), len(shot_list_validate)))
        print("Added bleed in shots to training and validation sets")
        # if num_d > 0:
        #     for i in range(num):
        #         s = shot_list_test.sample_single_class(True)
        #         shot_list_train.append(s)
        #         shot_list_validate.append(s)
        #         if conf['data']['bleed_in_remove_from_test']:
        #             shot_list_test.remove(s)
        # else:
        #     print('No disruptive shots in test set, [omit] bleed in')
        # if num_nd > 0:
        #     for i in range(num):
        #         s = shot_list_test.sample_single_class(False)
        #         shot_list_train.append(s)
        #         shot_list_validate.append(s)
        #         if conf['data']['bleed_in_remove_from_test']:
        #             shot_list_test.remove(s)
        # else:
        #     print('No nondisruptive shots in test set, [omit] bleed in')
    return shot_list_train, shot_list_validate, shot_list_test


def guarantee_preprocessed(conf, verbose=False):
    pp = Preprocessor(conf)
    if pp.all_are_preprocessed():
        if verbose:
            g.print_unique("shots already processed.")
        (shot_list_train, shot_list_validate,
         shot_list_test) = pp.load_shotlists()
    else:
        if verbose:
            g.print_unique("preprocessing all shots...")  # , end='')
        pp.clean_shot_lists()
        shot_list = pp.preprocess_all()
        shot_list.sort()
        shot_list_train, shot_list_test = shot_list.split_train_test(conf)
        # num_shots = len(shot_list_train) + len(shot_list_test)
        validation_frac = conf['training']['validation_frac']
        if validation_frac <= 0.05:
            if verbose:
                g.print_unique('Setting validation to a minimum of 0.05')
            validation_frac = 0.05
        shot_list_train, shot_list_validate = shot_list_train.split_direct(
            1.0-validation_frac, do_shuffle=True)
        pp.save_shotlists(shot_list_train, shot_list_validate, shot_list_test)
    shot_list_train, shot_list_validate, shot_list_test = apply_bleed_in(
        conf, shot_list_train, shot_list_validate, shot_list_test)
    if verbose:
        print_shot_list_sizes(shot_list_train, shot_list_validate,
                              shot_list_test)
        g.print_unique("...done")
    #    g.print_unique("...printing test shot list:")
    #    for s in shot_list_test:
    #       g.print_unique(str(s.number))
    return shot_list_train, shot_list_validate, shot_list_test
