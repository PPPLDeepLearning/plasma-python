from __future__ import print_function
import errno
import os
from functools import partial
import multiprocessing as mp
import sys
import time
import numpy as np
# import gadata
# from plasma.primitives.shots import ShotList

'''MDSplus references:
http://www.mdsplus.org/index.php?title=Documentation:Tutorial:RemoteAccess&open=76203664636339686324830207&page=Documentation%2FThe+MDSplus+tutorial%2FRemote+data+access+in+MDSplus
http://piscope.psfc.mit.edu/index.php/MDSplus_%26_python#Simple_example_of_reading_MDSplus_data
http://www.mdsplus.org/documentation/beginners/expressions.shtml
http://www.mdsplus.org/index.php?title=Documentation:Tutorial:MdsObjects&open=76203664636339686324830207&page=Documentation%2FThe+MDSplus+tutorial%2FThe+Object+Oriented+interface+of+MDSPlus
'''

'''TODO
- mapping to flux surfaces: its not always [0,1]!
- handling of 1D signals during preprocessing & normalization
- handling of 1D signals for feeding into RNN (convolutional layers)
- handling of missing data in shots?
- TEST
'''
try:
    from MDSplus import Connection
except ImportError:
    pass


def get_missing_value_array():
    return np.array([-1.0])


def makedirs_process_safe(dirpath):
    try:  # can lead to race condition
        os.makedirs(dirpath)
    except OSError as e:
        # File exists, and it's a directory, another process beat us to
        # creating this dir, that's OK.
        if e.errno == errno.EEXIST:
            pass
        else:
            # Our target dir exists as a file, or different error, reraise the
            # error!
            raise


def makedirdepth_process_safe(dirpath):
    try:  # can lead to race condition
        mkdirdepth(dirpath)
    except OSError as e:
        # File exists, and it's a directory, another process beat us to
        # creating this dir, that's OK.
        if e.errno == errno.EEXIST:
            pass
        else:
            # Our target dir exists as a file, or different error, reraise the
            # error!
            raise


def mkdirdepth(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def format_save_path(prepath, signal_path, shot_num):
    return prepath + signal_path + '/{}.txt'.format(shot_num)


def save_shot(shot_num_queue, c, signals, save_prepath, machine, sentinel=-1):
    missing_values = 0
    # if machine == 'd3d':
    #   reload(gadata) #reloads Gadata object with connection
    while True:
        shot_num = shot_num_queue.get()
        if shot_num == sentinel:
            break
        shot_complete = True
        for signal in signals:
            signal_path = signal.get_path(machine)
            save_path_full = signal.get_file_path(save_prepath, machine,
                                                  shot_num)
            success = False
            mapping = None
            if os.path.isfile(save_path_full):
                if os.path.getsize(save_path_full) > 0:
                    print('-', end='')
                    success = True
                else:
                    print('Signal {}, shot {} '.format(signal_path, shot_num),
                          'was downloaded incorrectly (empty file). ',
                          'Redownloading.')
            if not success:
                try:
                    try:
                        time, data, mapping, success = (
                            signal.fetch_data(machine, shot_num, c))
                        if not success:
                            print('No success shot {}, signal {}'.format(
                                shot_num, signal))
                    except Exception as e:
                        print(e)
                        sys.stdout.flush()
                        # missing_values += 1
                        print('Signal {}, shot {} missing. '.format(
                            signal_path, shot_num), 'Filling with zeros.')
                        success = False

                    if success:
                        data_two_column = np.vstack((np.atleast_2d(time),
                                                     np.atleast_2d(data))
                                                    ).transpose()
                        if mapping is not None:
                            mapping_two_column = np.vstack((
                                np.atleast_2d(time), np.atleast_2d(mapping))
                                                           ).transpose()
                            data_two_column = np.vstack(
                                (mapping_two_column, data_two_column))
                    makedirdepth_process_safe(save_path_full)
                    if success:
                        np.savetxt(save_path_full, data_two_column, fmt='%.5e')
                    else:
                        np.savetxt(save_path_full, get_missing_value_array(),
                                   fmt='%.5e')
                    print('.', end='')
                except BaseException:
                    print('Could not save shot {}, signal {}'.format(
                        shot_num, signal_path))
                    print('Warning: Incomplete!!!')
                    raise
            sys.stdout.flush()
            if not success:
                missing_values += 1
                shot_complete = False
        # only add shot to list if it was complete
        if shot_complete:
            print('saved shot {}'.format(shot_num))
            # complete_queue.put(shot_num)
        else:
            print('shot {} not complete. removing from list.'.format(shot_num))
    print('Finished with {} missing values total'.format(missing_values))
    return


def download_shot_numbers(shot_numbers, save_prepath, machine, signals):
    max_cores = machine.max_cores
    sentinel = -1
    fn = partial(save_shot, signals=signals, save_prepath=save_prepath,
                 machine=machine, sentinel=sentinel)
    # can only handle 8 connections at once :(
    num_cores = min(mp.cpu_count(), max_cores)
    queue = mp.Queue()
    # complete_shots = Array('i',zeros(len(shot_numbers)))# = mp.Queue()

    # mp.queue can't handle larger queues yet!
    assert len(shot_numbers) < 32000
    for shot_num in shot_numbers:
        queue.put(shot_num)
    for i in range(num_cores):
        queue.put(sentinel)
    connections = [Connection(machine.server) for _ in range(num_cores)]
    processes = [mp.Process(target=fn, args=(queue, connections[i]))
                 for i in range(num_cores)]
    print('running in parallel on {} processes'.format(num_cores))
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def download_all_shot_numbers(prepath, save_path, shot_list_files,
                              signals_full):
    max_len = 30000
    machine = shot_list_files.machine
    signals = []
    for sig in signals_full:
        if not sig.is_defined_on_machine(machine):
            print('Signal {} not defined on machine {} [omit]'.format(
                sig, machine))
        else:
            signals.append(sig)
    save_prepath = prepath + save_path + '/'
    shot_numbers, _ = shot_list_files.get_shot_numbers_and_disruption_times()
    # can only use queue of max size 30000
    shot_numbers_chunks = [shot_numbers[i:i+max_len]
                           for i in np.xrange(0, len(shot_numbers), max_len)]
    start_time = time.time()
    for shot_numbers_chunk in shot_numbers_chunks:
        download_shot_numbers(shot_numbers_chunk, save_prepath, machine,
                              signals)

    print('Finished downloading {} shots in {} seconds'.format(
        len(shot_numbers), time.time()-start_time))
