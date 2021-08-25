"""
The module composed in this file is designed to handle the processing/handling
and incorporation of electron cyclotron emission imaging data into the FRNN
disruption prediction software suite. It contains snippets from the rest of
the FRNN codebase, and therefore is partially redundant.
Jesse A Rodriguez, 06/28/2021
"""

import numpy as np
import matplotlib as mpl
#mpl.rcParams['figure.dpi']=10
import matplotlib.pyplot as plt
#plt.rc('font', family='tahoma')
#font = 1
#plt.rc('xtick', labelsize=font)
#plt.rc('ytick', labelsize=font)
import time
import sys
import os
import multiprocessing as mp
from functools import partial
import h5py
import scipy.signal
import math
try:
    import MDSplus as MDS
except ImportError:
    pass

################################################################################
## Utility Functions and Globals
################################################################################
def Fetch_ECEI_d3d(channel_path, shot_number, c = None, verbose = False):
    """
    Basic fetch ecei data function, uses MDSplus Connection objects and looks
    for data in all the locations we know of.

    Args:
        channel_path: str, path to save .txt file (channel folder, format 
                      "LFSxxxx")
        shot_number: int, DIII-D shot number
        c: MDSplus.Connection object. None by default
        verbose: bool, suppress print statements
    """
    channel = channel_path
    shot = str(int(shot_number))
    mds_fail_pd = False
    mds_fail_pd2 = False
    mds_fail_p = False
    mds_fail_t = False

    #ptdata2 method (seems to be most reliable)
    try:
        x_pd2 = c.get('dim_of(_s = ptdata2('+channel+','+shot+'))')
        y_pd2 = c.get('_s = ptdata2('+channel+','+shot+')')
    except Exception as e:
        if verbose:
            print(e)
        mds_fail_pd2 = True
        pass
    if not mds_fail_pd2:
        if x_pd2.shape[0] > 1:
            print('Data exists for shot '+shot+' in channel '+channel[-5:-1]+'.')
            return x_pd2, y_pd2, None, True
    
    #psuedo method
    try:
        x_p = c.get('dim_of(_s = psuedo('+channel+','+shot+'))')
        y_p = c.get('_s = psuedo('+channel+','+shot+')')
    except Exception as e:
        if verbose:
            print(e)
        mds_fail_p = True
        pass
    if not mds_fail_p:
        if x_p.shape[0] > 1:
            print('Data exists for shot '+shot+' in channel '+channel[-5:-1]+'.')
            return x_p, y_p, None, True
            
    #ptdata method
    try:
        x_pd = c.get('dim_of(_s = ptdata('+channel+','+shot+'))')
        y_pd = c.get('_s = ptdata('+channel+','+shot+')')
    except Exception as e:
        if verbose:
            print(e)
        mds_fail_pd = True
        pass
    if not mds_fail_pd:
        if x_pd.shape[0] > 1:
            print('Data exists for shot '+shot+' in channel '+channel[-5:-1]+'.')
            return x_pd, y_pd, None, True

    #tree method
    try:
        c.openTree(channel, shot)
        x_t = c.get('dim_of(_s = '+shot+')').data()
        y_t = c.get('_s = '+shot).data()
    except Exception as e:
        if verbose:
            print(e)
        mds_fail_t = True
        pass
    if not mds_fail_t:
        if x_t.shape[0] > 1:
            print('Data exists for shot '+shot+' in channel '+channel[-5:-1]+'.')
            return x_t, y_t, None, True

    print('Data DOES NOT exist for shot '+shot+' in channel '+channel[-5:-1]+'.')
    return None, None, None, False


def Download_Shot(shot_num_queue, c, n_shots, n_procs, channel_paths,\
                  sentinel = -1, verbose = False, d_sample = 1,\
                  try_again = False):
    """
    Accepts a multiprocessor queue of shot numbers and downloads/saves data for
    a single shot off the front of the queue.

    Args:
        shot_num_queue: multiprocessing queue object containing shot numbers
        c: MDSplus.Connection object
        n_shots: int, total number of shots to be processed
        n_proc: int, number of processes
        channel_paths: list containing savepaths to channel folders
        sentinel: sentinel value; -1 by default. Serves as the mechanism for
                  terminating the parallel program.
        verbose: bool, suppress print statements
        d_sample: int, downsample factor, MUST BE IN FORM 10^y
        try_again: bool, tells script to try and download signals that were
                   found to be missing in a prior run.
    """
    missing_shots = 0
    while True:
        shot_num = shot_num_queue.get()
        shots_left = shot_num_queue.qsize() - n_procs
        shots_progress = 100*(n_shots - shots_left)/n_shots
        shots_progress_next = 100*(n_shots + 1 - shots_left)/n_shots
        if shot_num == sentinel:
            break
        shot_complete = True
        chan_done = 0
        for channel_path in channel_paths:
            save_path = channel_path[:-9]+'{}.hdf5'.format(int(shot_num))
            channel = channel_path[-9:]

            success = False
            if os.path.isfile(save_path):
                if os.path.getsize(save_path) > 0:
                    f = h5py.File(save_path, 'r')
                    for key in f.keys():
                        if key == channel:
                            success = True
                        if key.startswith('missing') and key.endswith(channel)\
                           and not try_again:
                            success = True
                    f.close()
                else:
                    print('Shot {} '.format(int(shot_num)),'was downloaded \
                           incorrectly (empty file). Redownloading.')

            else:
                f = h5py.File(save_path, 'w')
                f.close()

            if not success:
                try:
                    try:
                        time, data, mapping, success = Fetch_ECEI_d3d(\
                                                channel_path[-9:], shot_num, c,\
                                                verbose)
                    except Exception as e:
                        print(e)
                        sys.stdout.flush()
                        print('Channel {}, shot {} missing, all mds commands \
                               failed.'.format(channel_path[-5:-1], shot_num))
                        success = False

                    if success:
                        data_two_column = np.vstack((time, data)).transpose()
                        if d_sample > 1:
                            n = int(math.log10(d_sample))
                            for _ in range(n):
                                data_two_column = scipy.signal.decimate(\
                                                  data_two_column, 10, axis = 0)
                        f = h5py.File(save_path, 'r+')
                        for key in f.keys():
                            if key.startswith('missing'):
                                if key[8:] == channel:
                                    del f[key]
                        dset = f.create_dataset(channel, data = data_two_column)
                        f.close()
                    else:
                        f = h5py.File(save_path, 'r+')
                        dsetk = 'missing_'+channel
                        already = False
                        for key in f.keys():
                            if key == dsetk:
                                f.close()
                                already = True
                        if not already:
                            dset = f.create_dataset(dsetk,\
                                                    data = np.array([-1.0]))
                            f.close()

                except BaseException:
                    print('Could not save channel {}, shot {}.'.format(\
                           channel_path[-5:-1], shot_num))
                    print('Warning: Incomplete!!!')
                    raise
            else:
                print('Channel {}, shot {} '.format(channel_path[-5:-1],\
                       int(shot_num)),'has already been downloaded.')
            sys.stdout.flush()
            if not success:
                missing_shots += 1
            chan_done += 1
            shot_prog = chan_done/160
            overall_prog = shots_progress +\
                           (shots_progress_next-shots_progress)*shot_prog
            print('Approximate download progress: {:.2f}%.'\
                  .format(overall_prog))

    print('Finished with {} channel signals missing.'.format(missing_shots))
    return
                         

def Download_Shot_List(shot_numbers, channel_paths, max_cores = 8,\
                       server = 'atlas.gat.com', verbose = False,\
                       d_sample = 1, try_again = False):
    """
    Accepts list of shots and downloads them in parallel

    Args:
        shot_numbers: list of integer shot numbers
        channel_paths: list of channel save path folders
        max_cores: int, max number of cores for parallelization
        server: MDSplus server, str. D3D server by default
        verbose: bool, suppress print statements
        d_sample: int, downsample factor, MUST BE IN FORM 10^y
        try_again: bool, tells script to try and download signals that were
                   found to be missing in a prior run.
    """
    sentinel = -1
    num_cores = min(mp.cpu_count(), max_cores)
    fn = partial(Download_Shot, n_shots = len(shot_numbers),\
                 n_procs = num_cores, channel_paths = channel_paths,\
                 sentinel = sentinel, verbose = verbose,\
                 d_sample = d_sample, try_again = try_again)
    queue = mp.Queue()
    assert len(shot_numbers) < 32000
    for shot_num in shot_numbers:
        queue.put(shot_num)
    for i in range(num_cores):
        queue.put(sentinel)

    connections = [MDS.Connection(server) for _ in range(num_cores)]
    processes = [mp.Process(target=fn, args = (queue, connections[i]))\
                 for i in range(num_cores)]
    print('Running in parallel on {} processes.'.format(num_cores))
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def Count_Missing(shot_list, shot_path, missing_path):
    """
    Accepts a shot list and a path to the shot files and produces an up-to-date
    list of all missing data and places it in missing_path. Automatically
    called after a download operation.

    Args:
        shot_list: 1-D numpy array of DIII-D shot numbers
        shot_path: path to folder containing shot files
        missing_path: folder for missing shot reports
    """
    min_shot = np.argmin(shot_list)
    max_shot = np.argmax(shot_list)
    num_shots = np.shape(shot_list)[0]*160
    num_shots_miss = 0
    print("Generating report for {} shots between shots {} and {}".format(\
           np.shape(shot_list)[0], int(shot_list[min_shot]),\
           int(shot_list[max_shot])))
    report = open(missing_path+'/missing_report_'+str(int(shot_list[min_shot]))\
                  +'-'+str(int(shot_list[max_shot]))+'.txt', mode = 'w',\
                  encoding='utf-8')
    report.write('Missing channel signals for download from shot {} to shot '\
                 '{}:\n'.format(int(shot_list[min_shot]),\
                                int(shot_list[max_shot])))
    for filename in os.listdir(shot_path):
        if filename.endswith('hdf5'):
            if int(filename[:-5]) >= shot_list[min_shot]\
            and int(filename[:-5]) <= shot_list[max_shot]:
                f = h5py.File(shot_path+'/'+filename, 'r')
                count = 0
                for key in f.keys():
                    if key.startswith('missing'):
                        count += 1
                        report.write('Channel '+key[-5:-1]+', shot #'+\
                                     filename[:-5]+'\n')
                        num_shots_miss +=1
                if count > 160:
                    print('Shot #'+filename[:-5]+' has more than 160 channels '\
                          'missing!')

    report.write('Missing channel signals for {} out of {} signals.'.\
                  format(num_shots_miss, num_shots))
    report.close()

    return (num_shots_miss, num_shots)


###############################################################################
## ECEI Class
###############################################################################
class ECEI:
    def __init__(self):
        """
        Initialize ECEI object by creating an internal list of channel keys.

        Args:
        """
        self.ecei_channels = []
        for i in range(20):
            for j in range(8):
                self.ecei_channels.append('"LFS{:02d}{:02d}"'.format(i+3,j+1))

    ###########################################################################
    ## Data Processing
    ###########################################################################
    def Generate_Missing_Report(self, shots, shot_1, clear_file, disrupt_file,\
                                save_path = os.getcwd()):
        """
        Accept a start shot and a number of clear shots and generate a verbose
        missing shot report for all shots in that range of the shot list files.

        Args:
            shots: int, number of non-disruptive shots you want to download
            shot_1: int, the shot number you want to start with
            clear_file: The path to the clear shot list
            disrupt_file: The path to the disruptive shot list
            save_path: location where the shot hdf5 files will be stored,
                       current directory by default
        """
        clear_shots = np.loadtxt(clear_file)
        disrupt_shots = np.loadtxt(disrupt_file)

        first_c = False
        first_d = False
        i = 0
        while not first_c:
            if clear_shots[i,0] >= shot_1:
                start_c = i
                first_c = True
            i += 1
        i = 0
        while not first_d:
            if disrupt_shots[i,0] >= shot_1:
                start_d = i
                first_d = True
            i += 1

        if start_c + shots > clear_shots.shape[0]-1:
            shots = clear_shots.shape[0] - start_c - 1

        shot_list = np.array([clear_shots[start_c,0]])
        for i in range(shots-1):
            shot_list = np.append(shot_list, [clear_shots[i+start_c+1,0]])

        last = False
        no_disrupt = False
        i = start_d
        while not last:
            if disrupt_shots[i,0] >= clear_shots[start_c+shots-1,0]:
                end_d = i
                last = True
            i += 1
            if i >= disrupt_shots.shape[0]:
                no_disrupt = True
                last = True

        if not no_disrupt:
            for i in range(end_d - start_d + 1):
                shot_list = np.append(shot_list, [disrupt_shots[i+start_d,0]])
            
        channel_paths = []
        for i in range(len(self.ecei_channels)):
            channel_path = os.path.join(save_path, self.ecei_channels[i])
            channel_paths.append(channel_path)
        #Missing shots directory
        missing_path = os.path.join(save_path, 'missing_shot_info')
        if not os.path.exists(missing_path):
            os.mkdir(missing_path)

        missed = Count_Missing(shot_list, save_path, missing_path)

        return


    def Clean_Signals(self, save_path = os.getcwd()):
        """
        Removes all signal files in the save_path directory.
        """
        check = input("WARNING: this function will delete ALL signal files in \
                the "+"designated save path. Type 'yes' to continue, anything \
                else to cancel.\n")
        if check == 'yes':
            for filename in os.listdir(save_path):
                if filename.endswith('hdf5'):
                    shot = os.path.join(save_path, filename)
                    os.remove(shot)


    def Clean_Missing_Signals(self, missing_path, save_path = os.getcwd()):
        """
        Removes all signal files with all channels missing in the save_path 
        directory.
        """
        check = input("WARNING: this function will delete ALL signal files in "\
                      "the designated save path which have all channel signals"\
                      " missing. Type 'yes' to continue, anything else to "\
                      "cancel.\n")
        report = open(missing_path+'/AllChannelsMissing.txt', mode = 'w',\
                  encoding='utf-8')
        if check == 'yes':
            for filename in os.listdir(save_path):
                if filename.endswith('hdf5'):
                    shot = os.path.join(save_path, filename)
                    f = h5py.File(shot, 'r')
                    count = 0
                    for key in f.keys():
                        if key.startswith('missing'):
                            count += 1
                    if count == 160:
                        f.close()
                        if os.path.getsize(shot) <= 342289:
                            report.write(filename[:-5]+"\n")
                            os.remove(shot)
                    else:
                        f.close()

        report.close()



    def Clean_Missing_Signal(self, shot_file):
        """
        Removes a single shot file if it has all channel signals missing.
        """
        shot = os.path.join(os.getcwd(), shot_file)
        f = h5py.File(shot, 'r')
        count = 0
        for key in f.keys():
            if key.startswith('missing'):
                count += 1
        if count == 160:
            f.close()
            check = input("You are about to delete "+shot+". Are "+\
                          "you sure about that?\n")
            if check == 'yes':
                os.remove(shot)
        else:
            f.close()


    def Clean_Missing_Signal_List(self, shots):
        """
        Removes shot files in a list if they have all channel signals missing.
        """
        for s in shots:
            shot_file = str(s)+".hdf5"
            shot = os.path.join(os.getcwd(), shot_file)
            f = h5py.File(shot, 'r')
            count = 0
            for key in f.keys():
                if key.startswith('missing'):
                    count += 1
            if count == 160:
                f.close()
                check = input("You are about to delete "+shot+". Are "+\
                              "you sure about that?\n")
                if check == 'yes':
                    os.remove(shot)
            else:
                f.close()


    def Generate_Missing_Report_Concise(self, todays_date,\
            data_path = os.getcwd(), output_path = os.getcwd()):
        """
        Creates a report of missing data in a more readable format.

        Args:
            todays_date: str, todays date in a readable, filename-friendly
                         format, like "MM-DD-YYYY"
            data_path: str, path where data files are stored
            output_path: str, path where the report will go
        """
        # Collect necessary information.
        shot_count = 0
        none_missing = 0
        all_missing = 0
        one_missing = 0
        eight_missing = 0
        sixteen_missing = 0
        sixteen_to_all_missing = 0
        one_to_sixteen_missing = 0
        missing_by_chan = {}
        all_missing_list = []
        some_missing_list = []
        full_shot_list = []
        file_list = os.listdir(data_path)
        num_shots = len(file_list)
        print("Generating concise report for the {} shots in "\
              .format(int(num_shots))+data_path)
        t_b = time.time()
        for filename in file_list:
            if filename.endswith('hdf5'):
                f = h5py.File(data_path+'/'+filename, 'r')
                miss_count = 0
                for key in f.keys():
                    if key[-9:] not in missing_by_chan:
                        missing_by_chan[key[-9:]] = 0
                    if key.startswith('missing'):
                        miss_count += 1
                        missing_by_chan[key[-9:]] += 1
                if miss_count == 160:
                    all_missing += 1
                    for key in f.keys():
                        missing_by_chan[key[-9:]] -= 1
                    all_missing_list.append(int(filename[:-5]))
                elif miss_count == 1:
                    one_missing += 1
                elif miss_count == 8:
                    eight_missing += 1
                elif miss_count == 16:
                    sixteen_missing += 1
                elif miss_count > 0 and miss_count <= 16:
                    one_to_sixteen_missing += 1
                    some_missing_list.append(int(filename[:-5]))
                elif miss_count > 16 and miss_count < 160:
                    sixteen_to_all_missing += 1
                    some_missing_list.append(int(filename[:-5]))
                elif miss_count == 0:
                    none_missing += 1
                    full_shot_list.append(int(filename[:-5]))
                shot_count += 1
                f.close()
                if shot_count%10 == 0:
                    print("{:.2f}% of the way through collecting missing shot "\
                          "info.".format(shot_count/num_shots*100))

        t_e = time.time()
        T = t_e-t_b

        print("Finished collecting info in {} seconds.".format(T))

        # Write report
        report = open(output_path+'/missing_signal_report_'+todays_date+'.txt',\
                      'w')
        report.write('This missing shot report was generated using the \
                     contents of '+output_path+' on '+todays_date+'.\n\n')
        report.write('Number of shots with NO channels missing: {}\n'.format(\
                     int(none_missing)))
        report.write('Number of shots with ALL channels missing: {}\n'.format(\
                     int(all_missing)))
        report.write('Number of shots with just one channel missing: {}\n'\
                     .format(int(one_missing)))
        report.write('Number of shots with 8 channels missing: {}\n'.format(\
                     int(eight_missing)))
        report.write('Number of shots with 16 channels missing: {}\n'.format(\
                     int(sixteen_missing)))
        report.write('Number of shots with 2 to 15 channels missing: {}\n'\
                     .format(int(one_to_sixteen_missing)))
        report.write('Number of shots with 17 to 159 channels missing: {}\n\n'\
                     .format(int(sixteen_to_all_missing)))
        report.write('Missing signal distribution by channel in shots with '+\
                     'fewer than 160 channels missing:\n')
        missing_chan_tot = 0
        most_miss = 0
        for key in missing_by_chan:
            missing_chan_tot += missing_by_chan[key]
            if missing_by_chan[key] > most_miss:
                most_miss = missing_by_chan[key]

        for i in range(20):
            for j in range(8):
                key = '"LFS{:02d}{:02d}"'.format(i+3, j+1)
                bar_length = int(missing_by_chan[key]/most_miss*50)
                bar = '█'*bar_length
                report.write('Channel {:02d}{:02d}: '.format(i+3, j+1)+\
                        str(int(missing_by_chan[key]))+' | '+bar+'\n')

        report.close()

        all_missing_list = np.sort(all_missing_list)
        some_missing_list = np.sort(some_missing_list)
        full_shot_list = np.sort(full_shot_list)

        np.savetxt(output_path+'/all_channels_missing_list.txt',\
                   all_missing_list, fmt='%i')
        np.savetxt(output_path+'/some_channels_missing_list.txt',\
                   some_missing_list, fmt='%i')
        np.savetxt(output_path+'/no_channels_missing_list.txt', full_shot_list,\
                   fmt='%i')


    def Generate_Quality_Report(self, todays_date, data_path, disrupt_list,\
                                shots_of_interest, shotlist_name, output_path =\
                                os.getcwd()):
        """
        Create a report that checks shots in shots_of_interest for NaNs, as
        well as for cases in which the data collection ceases before t_disrupt.
        If the shots are missing channels, the report will give the number of
        channels missed per shot.

        Args:
            todays_date: str, todays date in a readable, filename-friendly
                         format, like "MM-DD-YYYY"
            data_path: str, path where data files are stored.
            disrupt_list: numpy array, shot list that contains the disruptive
                          shots of interest.
            shots_of_interest: numpy array, shot list that contains the shots
                               you would like to check.
            shotlist_name: str, name describing the shotlist of interest.
            output_path: str, path where the report will go.
        """
        print("Generating a data quality report for {} shots of interest in "\
              .format(int(len(shots_of_interest)))+data_path)
        t_b = time.time()

        contains_NaN = {}
        ends_before_t_disrupt = {}
        missing_chans = {}
        low_std_dev = {}

        count = 0
        files = os.listdir(data_path)
        num_files = len(files)
        for filename in files:
            if filename.endswith('hdf5'):
                count += 1
                shot_no = int(filename[:-5])
                if shot_no in shots_of_interest:
                    f = h5py.File(data_path+'/'+filename, 'r')
                    keys = f.keys()
                    # First we check for NaNs
                    for key in keys:
                        data = np.asarray(f.get(key))
                        if np.any(np.isnan(data)):
                            if shot_no not in contains_NaN:
                                contains_NaN[shot_no] = []
                            contains_NaN[shot_no].append(key[-5:-1])
                        # Check to make sure 'something' happens
                        if not key.startswith('missing'):
                            sig = np.sqrt(np.var(data[:,1]))
                            if sig < 0.001:
                                if shot_no not in low_std_dev:
                                    low_std_dev[shot_no] = []
                                low_std_dev[shot_no].append(key[-5:-1])
                        # Next, missing channels
                        if key.startswith('missing'):
                            if shot_no not in missing_chans:
                                missing_chans[shot_no] = []
                            missing_chans[shot_no].append(key[-5:-1])
                        # Now we check if data is collected up to t_disrupt
                        if shot_no in disrupt_list[:,0] and\
                           not key.startswith('missing'):
                            i_disrupt=np.where(disrupt_list[:,0]==shot_no)[0][0]
                            t_max = np.max(data[:,0])
                            t_disrupt = disrupt_list[i_disrupt,1]
                            if t_max < t_disrupt:
                                if shot_no not in ends_before_t_disrupt:
                                    ends_before_t_disrupt[shot_no] = []
                                ends_before_t_disrupt[shot_no].append(key[-5:-1])
                    print("{:2f}% of the way through shot files"\
                          .format(count/num_files*100))
                    f.close()

        t_e = time.time()
        T = t_e-t_b

        print("Finished collecting info in {} seconds.".format(T))

        # Write report
        report = open(output_path+'/data_quality_report_'+todays_date+'.txt',\
                      'w')
        report.write('This data quality report was generated using the '\
                     'contents of '+output_path+'\non '+todays_date+', using '\
                     'a shotlist named "'+shotlist_name+'".\n\n')

        report.write('Number of shots with NaNs present: {}\n'.format(\
                     int(len(contains_NaN))))
        if len(contains_NaN) > 0:
            for shot in contains_NaN:
                report.write('Shot {} Contains NaNs in the following '\
                             'channels:\n'.format(shot))
                count = 0
                for i in range(len(contains_NaN[shot])):
                    count += 1
                    if count%10 == 0:
                        report.write(contains_NaN[shot][i]+',\n')
                    else:
                        report.write(contains_NaN[shot][i]+', ')
                report.write('\n')

        report.write('_'*80)
        report.write('\n\n')
        report.write('Number of shots that cease data collection before '\
                     't_disrupt: {}\n'.format(int(len(ends_before_t_disrupt))))
        if len(ends_before_t_disrupt) > 0:
            for shot in ends_before_t_disrupt:
                report.write('Shot {} stops short of t_disrupt in the '\
                             'following channels:\n'.format(shot))
                count = 0
                for i in range(len(ends_before_t_disrupt[shot])):
                    count += 1
                    if count%10 == 0:
                        report.write(ends_before_t_disrupt[shot][i]+',\n')
                    else:
                        report.write(ends_before_t_disrupt[shot][i]+', ')
                report.write('\n')

        report.write('_'*80)
        report.write('\n\n')
        report.write('Number of shots that have a standard deviation which is '\
                     'smaller than 1 mV: {}\n'.format(int(len(low_std_dev))))
        if len(low_std_dev) > 0:
            for shot in low_std_dev:
                report.write('Shot {} has a std. dev less than 1 mV in the '\
                             'following channels:\n'.format(shot))
                count = 0
                for i in range(len(low_std_dev[shot])):
                    count += 1
                    if count%10 == 0:
                        report.write(low_std_dev[shot][i]+',\n')
                    else:
                        report.write(low_std_dev[shot][i]+', ')
                report.write('\n')

        report.write('_'*80)
        report.write('\n\n')
        report.write('Number of shots with missing channels: {}\n'.format(\
                     int(len(missing_chans))))
        if len(missing_chans) > 0:
            for shot in missing_chans:
                report.write('Shot {} is missing data in the following '\
                             'channels:\n'.format(shot))
                count = 0
                for i in range(len(missing_chans[shot])):
                    count += 1
                    if count%10 == 0:
                        report.write(missing_chans[shot][i]+',\n')
                    else:
                        report.write(missing_chans[shot][i]+', ')
                report.write('\n')

        report.close()


    ###########################################################################
    ## Visualization
    ###########################################################################
    def Single_Shot_Plot(self, shot, data_path, save_dir = os.getcwd(),\
                         show = False):
        """
        Plot voltage traces for a single shot, saves plot as a .pdf

        Args:
            shot: int, shot number
            data_path: str, path to ECEI data
            save_dir: str, directory for output plot image
            shot: bool, determines whether output is shown right away
        """
        shot_file = data_path+'/'+str(int(shot))+'.hdf5'
        f = h5py.File(shot_file, 'r')
        fig = plt.figure()
        gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0)
        ax = gs.subplots(sharex='col')
        count = 0
        plot_no = 0
        for channel in self.ecei_channels:
            count += 1
            row = plot_no//5
            col = plot_no%5
            if channel in f.keys():
                data = f.get(channel)
                ax[row,col].plot(data[:,0], data[:,1],\
                                 label = 'YY = '+channel[-3:-1],\
                                 linewidth = 0.4, alpha = 0.8)
            if count%8 == 0:
                plot_no += 1
                XX = count//8 + 2
                title = 'XX = {:02d}'.format(XX)
                ax[row,col].set_title(title, fontsize = 5)
                #ax[row,col].legend(prop={'size': 2.75})
                ax[row,col].tick_params(width = 0.3)

        fig.suptitle('Shot #{}'.format(int(shot)), fontsize = 10)
        for axs in ax.flat:
            axs.set_xlabel('Time (ms)', fontsize = 5)
            axs.set_ylabel('ECEi Voltage (V)', fontsize = 5)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for axs in ax.flat:
            axs.label_outer()
            axs.tick_params(axis='x', labelsize = 5)
            axs.tick_params(axis='y', labelsize = 5)

        labels = []
        for i in range(8):
            labels.append('YY = {:2d}'.format(i+1))
        fig.legend(labels=labels, loc="lower center", ncol=8, prop={'size': 5.5})

        if show: 
            fig.show()

        fig.savefig(save_dir+'/Shot_{}.pdf'.format(int(shot)))


    def Generate_Txt(self, shot, channel, save_dir = os.getcwd()):
        """
        Get a .txt file out for signal data in a readable format for a single
        channel.

        Args:
            shot: int, shot number
            channel: str, format "XXYY", 03<=XX<=22, 01<=YY<=08, designates
                     channel
            save_dir: str, directory where shot files are stored
        """
        shot_s = str(shot)
        f = h5py.File(save_dir+'/'+shot_s+'.hdf5', 'r')
        data = np.asarray(f.get('"LFS'+channel+'"'))
        np.savetxt(save_dir+'/'+shot_s+'_chan'+channel+'.txt', data)
        f.close()

        return

    
    def Generate_Txt_Interactive(self, save_dir = os.getcwd()):
        """
        Get a .txt file out for reading signal data, accepts input from command
        line.

        Args:
            save_dir: str, directory where shot files are stored
        """
        shot = int(input("Which shot? Enter an integer.\n"))
        channel = input("Which channel? format 'XXYY', 03<=XX<=22, 01<=YY<=08"\
                        ".\n")

        self.Generate_Txt(shot, channel, save_dir)

        return


    ###########################################################################
    ## Data Acquisition
    ###########################################################################
    def Acquire_Shots_D3D(self, shot_numbers, save_path = os.getcwd(),\
                          max_cores = 8, verbose = False, chan_lowlim = 3,\
                          chan_uplim = 22, d_sample = 1, try_again = False):
        """
        Accepts a list of shot numbers and downloads the data, saving them into
        folders corresponding to the individual channels. Returns nothing. 
        Shots are saved in hdf5 format, downsampling is done BEFORE saving. 
        Each channel is labelled within its own dataset in the hdf5 file, where 
        the label is the channel name/MDS point name, e.g. '"LFSXXYY"'. If data
        is not found, labels are 'missing_"LFSXXYY"' with [-1.0] as the dataset.

        Args:
            shot_numbers: 1-D numpy array of integers, DIII-D shot numbers
            save_path: location where the channel folders will be stored,
                       current directory by default
            max_cores: int, max # of cores to carry out download tasks
            verbose: bool, suppress most print statements
            chan_lowlim: int, lower limit of subset of channels to download
            chan_uplim: int, upper limit of subset of channels to download
            d_sample: int, downsample factor, MUST BE IN FORM 10^y
            try_again: bool, tells script to try and download signals that were
                       found to be missing in a prior run.
        """
        t_b = time.time()
        # Construct channel save paths and create them if needed.
        channel_paths = []
        for i in range(len(self.ecei_channels)):
            XX = int(self.ecei_channels[i][-5:-3])
            if XX >= chan_lowlim and XX <= chan_uplim:
                channel_path = os.path.join(save_path, self.ecei_channels[i])
                channel_paths.append(channel_path)
        #Missing shots directory
        missing_path = os.path.join(save_path, 'missing_shot_info')
        if not os.path.exists(missing_path):
            os.mkdir(missing_path)

        try:
            c = MDS.Connection('atlas.gat.com')
        except Exception as e:
            print(e)
            return False

        Download_Shot_List(shot_numbers, channel_paths, max_cores = max_cores,\
                           server = 'atlas.gat.com', verbose = verbose,\
                           d_sample = d_sample, try_again = try_again)

        missed = Count_Missing(shot_numbers, save_path, missing_path)

        t_e = time.time()
        T = t_e-t_b

        print("Downloaded {} out of {} signals in {} seconds."\
              .format(missed[1]-missed[0], missed[1], T))

        return


    def Acquire_Shot_Sequence_D3D(self, shots, shot_1, clear_file,\
                                  disrupt_file, save_path = os.getcwd(),\
                                  max_cores = 8, verbose = False,\
                                  chan_lowlim = 3, chan_uplim = 22,\
                                  d_sample = 1, try_again = False):
        """
        Accepts a desired number of non-disruptive shots, then downloads all
        shots in our labelled database up to the last non-disruptive shot.
        Returns nothing. Shots are saved in hdf5 format, downsampling is done
        BEFORE saving. Each channel is labelled within its own dataset in the
        hdf5 file, where the label is the channel name/MDS point name, e.g.
        '"LFSXXYY"'. If data is not found, labels are 'missing_"LFSXXYY"' with 
        [-1.0] as the dataset.

        Args:
            shots: int, number of non-disruptive shots you want to download
            shot_1: int, the shot number you want to start with
            clear_file: The path to the clear shot list
            disrupt_file: The path to the disruptive shot list
            save_path: location where the channel folders will be stored,
                       current directory by default
            max_cores: int, max # of cores to carry out download tasks
            verbose: bool, suppress some exception info
            chan_lowlim: int, lower limit of subset of channels to download
            chan_uplim: int, upper limit of subset of channels to download
            d_sample: int, downsample factor, MUST BE IN FORM 10^y
            try_again: bool, tells script to try and download signals that were
                       found to be missing in a prior run.
        """
        clear_shots = np.loadtxt(clear_file)
        disrupt_shots = np.loadtxt(disrupt_file)

        first_c = False
        first_d = False
        i = 0
        while not first_c:
            if clear_shots[i,0] >= shot_1:
                start_c = i
                first_c = True
            i += 1
        i = 0
        while not first_d:
            if disrupt_shots[i,0] >= shot_1:
                start_d = i
                first_d = True
            i += 1

        if start_c + shots > clear_shots.shape[0]-1:
            shots = clear_shots.shape[0] - start_c - 1

        shot_list = np.array([clear_shots[start_c,0]])
        for i in range(shots-1):
            shot_list = np.append(shot_list, [clear_shots[i+start_c+1,0]])

        last = False
        no_disrupt = False
        i = start_d
        while not last:
            if disrupt_shots[i,0] >= clear_shots[start_c+shots-1,0]:
                end_d = i
                last = True
            i += 1
            if i >= disrupt_shots.shape[0]:
                no_disrupt = True
                last = True

        if not no_disrupt:
            for i in range(end_d - start_d + 1):
                shot_list = np.append(shot_list, [disrupt_shots[i+start_d,0]])

        self.Acquire_Shots_D3D(shot_list, save_path, max_cores, verbose,\
                               chan_lowlim, chan_uplim, d_sample, try_again)

        return


    def Fetch_Shot(self, shot_number, verbose = False):
        """
        Fetch shot data from MDSplus server directly as numpy arrays. Returns
        a 1D time array and a 2D data array with shape (t_steps, 160), where
        each column is the signal data from one channel, along with None in
        place of a mapping and True to indicate success. Missing channels are
        padded with zeros.

        Args:
            shot_number: int, shot number
            verbose: bool, determines if MDS exceptions are printed
        """
        no_time = True
        idx = 0
        while no_time:
            t, d, mapping, success = Fetch_ECEI_d3d(self.ecei_channels[idx],\
                                                    shot_number, verbose = verbose)
            if success:
                time = np.asarray(t)
                no_time = False
            idx += 1
            if idx >= 160:
                return None, None, None, False

        no_data = True
        for channel in self.ecei_channels:
            t, d, mapping, success = Fetch_ECEI_d3d(channel, shot_number,\
                                                    verbose = verbose)
            if success:
                if no_data:
                    data = np.asarray(d).reshape((time.shape[0],1))
                    no_data = False
                else:
                    data = np.append(data, d, axis = 1)
            else:
                if no_data:
                    data = np.zeros((time.shape[0],1))
                    no_data = False
                else:
                    d = np.zeros((time.shape[0],1))
                    data = np.append(data, d, axis = 1)

        return time, data, None, True


