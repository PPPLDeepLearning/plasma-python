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


    # def get_signals_and_times_from_file(self,shot,t_disrupt):
    #     valid = True
    #     t_min = -1
    #     t_max = np.Inf
    #     t_thresh = -1
    #     signals = []
    #     times = []
    #     conf = self.conf

    #     disruptive = t_disrupt >= 0

    #     signal_prepath = conf['paths']['signal_prepath']
    #     signals_dirs = concatenate_sublists(conf['paths']['signals_dirs'])
    #     current_index = conf['data']['current_index']
    #     current_thresh = conf['data']['current_thresh']
    #     current_end_thresh = conf['data']['current_end_thresh']
    #     for (i,dirname) in enumerate(signals_dirs):
    #         data = np.loadtxt(get_individual_shot_file(signal_prepath+dirname + '/',shot))
    #         t = data[:,0]
    #         sig = data[:,1]
    #         t_min = max(t_min,t[0])
    #         t_max = min(t_max,t[-1])
    #         if i == current_index:
    #             #throw out shots that never reach curren threshold
    #             if not (np.any(abs(sig) > current_thresh)):
    #                 valid = False
    #                 print('Shot {} does not exceed current threshold... invalid.'.format(shot))
    #             else:
    #                 #begin shot once current reaches threshold
    #                 index_thresh = np.argwhere(abs(sig) > current_thresh)[0][0]
    #                 t_thresh = t[index_thresh]
    #                 #end shot once current drops below current_end_thresh
    #                 if not disruptive:
    #                     acceptable_region = np.zeros_like(sig,dtype=bool)
    #                     acceptable_region[index_thresh:] = True
    #                     index_end_thresh = np.argwhere(np.logical_and(abs(sig) < current_end_thresh,acceptable_region))[0][0]
    #                     t_end_thresh = t[index_end_thresh]
    #                     assert(t_thresh < t_end_thresh < t_max)
    #                     t_max = t_end_thresh
    #         signals.append(sig)
    #         times.append(t)
    #     if not valid:
    #         t_thresh = t_min
    #     assert(t_thresh >= t_min)
    #     assert(t_disrupt <= t_max)
    #     if disruptive:
    #         assert(t_thresh < t_disrupt)
    #         t_max = t_disrupt
    #     t_min = t_thresh

    #     return signals,times,t_min,t_max,t_thresh,valid



    # def cut_and_resample_signals(self,times,signals,t_min,t_max,is_disruptive):
    #     dt = self.conf['data']['dt']
    #     T_max = self.conf['data']['T_max']

    #     #resample signals
    #     signals_processed = []
    #     assert(len(signals) == len(times) and len(signals) > 0)
    #     tr = 0
    #     for i in range(len(signals)):
    #         tr,sigr = cut_and_resample_signal(times[i],signals[i],t_min,t_max,dt)
    #         signals_processed.append(sigr)

    #     signals = signals_processed
    #     signals = np.column_stack(signals)

    #     if is_disruptive:
    #         ttd = max(tr) - tr
    #         ttd = np.clip(ttd,0,T_max)
    #     else:
    #         ttd = T_max*np.ones_like(tr)
    #     ttd = np.log10(ttd + 1.0*dt/10)
    #     return signals,ttd


    def get_shot_list_path(self):
        return self.conf['paths']['base_path'] + '/processed_shotlists/' + self.conf['paths']['data'] + '/shot_lists.npz'

    def load_shotlists(self,conf):
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
