'''
#########################################################
This file containts classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
from os import listdir,remove
import time
import sys
import os

import numpy as np
import pathos.multiprocessing as mp

from plasma.utils.processing import *
from plasma.primitives.shots import ShotList
from plasma.utils.downloading import mkdirdepth

class Preprocessor(object):

    def __init__(self,conf):
        self.conf = conf


    def clean_shot_lists(self):
        shot_list_dir = self.conf['paths']['shot_list_dir']
        paths = [os.path.join(shot_list_dir, f) for f in listdir(shot_list_dir) if os.path.isfile(os.path.join(shot_list_dir, f))]
        for path in paths:
            self.clean_shot_list(path)


    def clean_shot_list(self,path):
        data = np.loadtxt(path)
        ending_idx = path.rfind('.')
        new_path = append_to_filename(path,'_clear')
        if len(np.shape(data)) < 2:
            #nondisruptive
            nd_times = -1.0*np.ones_like(data)
            data_two_column = np.vstack((data,nd_times)).transpose()
            np.savetxt(new_path,data_two_column,fmt = '%d %f')
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
        train_frac = conf['training']['train_frac']
        use_shots_train = int(round(train_frac*use_shots))
        use_shots_test = int(round((1-train_frac)*use_shots))
#        print(use_shots_train)
#        print(use_shots_test) #each print out 100,000
        # if len(shot_files_test) > 0:
        #     return self.preprocess_from_files(shot_list_dir,shot_files_train,machines_train,use_shots_train) + \
        #        self.preprocess_from_files(shot_list_dir,shot_files_test,machines_train,use_shots_test)
        # else:
        return self.preprocess_from_files(shot_files_all,use_shots)


    def preprocess_from_files(self,shot_files,use_shots):
        #all shots, including invalid ones
        all_signals = self.conf['paths']['all_signals'] 
        shot_list = ShotList()
        shot_list.load_from_shot_list_files_objects(shot_files,all_signals)
        shot_list_picked = shot_list.random_sublist(use_shots)

        #empty
        used_shots = ShotList()

        use_cores = max(1,mp.cpu_count()-2)
        pool = mp.Pool(use_cores)
        print('running in parallel on {} processes'.format(pool._processes))
        start_time = time.time()
        for (i,shot) in enumerate(pool.imap_unordered(self.preprocess_single_file,shot_list_picked)):
        #for (i,shot) in enumerate(map(self.preprocess_single_file,shot_list_picked)):
            sys.stdout.write('\r{}/{}'.format(i,len(shot_list_picked)))
            used_shots.append_if_valid(shot)

        pool.close()
        pool.join()
        print('Finished Preprocessing {} files in {} seconds'.format(len(shot_list_picked),time.time()-start_time))
        print('Omitted {} shots of {} total.'.format(len(shot_list_picked) - len(used_shots),len(shot_list_picked)))
        print('{}/{} disruptive shots'.format(used_shots.num_disruptive(),len(used_shots)))
        return used_shots 

    def preprocess_single_file(self,shot):
        processed_prepath = self.conf['paths']['processed_prepath']
        recompute = self.conf['data']['recompute']
        # print('({}/{}): '.format(num_processed,use_shots))
        if recompute or not shot.previously_saved(processed_prepath):
            shot.preprocess(self.conf)
            shot.save(processed_prepath)

        else:
            try:
                shot.restore(processed_prepath,light=True)
                sys.stdout.write('\r{} exists.'.format(shot.number))
            except:
                shot.preprocess(self.conf)
                shot.save(processed_prepath)
                sys.stdout.write('\r{} exists but corrupted, resaved.'.format(shot.number))
        shot.make_light()
        return shot 


    def get_individual_channel_dirs(self):
        signals_dirs = self.conf['paths']['signals_dirs']

    def get_shot_list_path(self):
        return self.conf['paths']['base_path'] + '/processed_shotlists/' + self.conf['paths']['data'] + '/shot_lists.npz'

    def load_shotlists(self):
        path = self.get_shot_list_path()
        data = np.load(path)
        shot_list_train = data['shot_list_train'][()]
        shot_list_validate = data['shot_list_validate'][()]
        shot_list_test = data['shot_list_test'][()]
        return shot_list_train,shot_list_validate,shot_list_test


    def save_shotlists(self,shot_list_train,shot_list_validate,shot_list_test):
        path = self.get_shot_list_path()
        mkdirdepth(path)
        np.savez(path,shot_list_train=shot_list_train,shot_list_validate=shot_list_validate,shot_list_test=shot_list_test)


def guarantee_preprocessed(conf):
    pp = Preprocessor(conf)
    if pp.all_are_preprocessed():
        print("shots already processed.")
        shot_list_train,shot_list_validate,shot_list_test = pp.load_shotlists()
    else:
        print("preprocessing all shots",end='')
        pp.clean_shot_lists()
        shot_list = pp.preprocess_all()
        shot_list.sort()
        shot_list_train,shot_list_test = shot_list.split_train_test(conf)
        num_shots = len(shot_list_train) + len(shot_list_test)
        validation_frac = conf['training']['validation_frac']
        if validation_frac <= 0.05:
            print('Setting validation to a minimum of 0.05')
            validation_frac = 0.05
        shot_list_train,shot_list_validate = shot_list_train.split_direct(1.0-validation_frac,do_shuffle=True)
        pp.save_shotlists(shot_list_train,shot_list_validate,shot_list_test)
    print('validate: {} shots, {} disruptive'.format(len(shot_list_validate),shot_list_validate.num_disruptive()))
    print('training: {} shots, {} disruptive'.format(len(shot_list_train),shot_list_train.num_disruptive()))
    print('testing: {} shots, {} disruptive'.format(len(shot_list_test),shot_list_test.num_disruptive()))
    print("...done")
    return shot_list_train,shot_list_validate,shot_list_test

