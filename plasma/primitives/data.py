from __future__ import division
import numpy as np
import sys
import os
import re
import h5py

from scipy.interpolate import UnivariateSpline
from plasma.utils.processing import get_individual_shot_file
from plasma.utils.downloading import get_missing_value_array
from plasma.utils.hashing import myhash
from plasma.utils.ECEI import ECEI

# class SignalCollection:
#   """GA Data Obj"""
#   def __init__(self,signal_descriptions,signal_paths):
#       self.signals = []
#       for i in range(len(signal_paths))
#           self.signals.append(Signal(signal_descriptions[i],signal_paths[i]))

try:
    from MDSplus import Connection
except ImportError:
    pass


###############################################################################
# Parent Signal Class
###############################################################################
class Signal(object):
    """
    A Signal object is a wrapper for a single signal. It is used to fetch data
    remotely and load locally stored data, as well as to return useful
    information about the data which is used during processing and model
    training.

    Attributes:
        description: str, full name of signal, e.g. "Plasma density"
        paths: list of str, MDSplus point-names, must correspond in index to
               the Machine objects in self.machines
        machines: list of Machine objects that the signal is defined on.
        causal_shifts: list of floats; the causal shift needed to be sure the
                       signal is not utilizing future data.
        is_ip: bool, True if signal is plasma current
        num_channels: int, number of data collection channels. Used in profile
                      signals and two dimensional signals
        normalize: bool, True if signal is to be normalized (?)
        data_avail_tolerances: list of floats, value in s of the maximum 
                               allowable time between cessation of data 
                               collection and t_disrupt for each machine
        is_strictly_positive: bool, True if signal is strictly positive
        mapping_paths: list of str, MDSplus mapping paths
    """
    def __init__(self, description, paths, machines, tex_label=None,
                 causal_shifts=None, is_ip=False, normalize=True,
                 data_avail_tolerances=None, is_strictly_positive=False,
                 mapping_paths=None):
        assert len(paths) == len(machines)
        self.description = description
        self.paths = paths
        self.machines = machines
        if causal_shifts is None:
            causal_shifts = [0 for m in machines]
        self.causal_shifts = causal_shifts  # causal shift in ms -> (JAR) the causal shifts appear to be supplied in s in signals.py, NOT ms
        self.is_ip = is_ip
        self.num_channels = 1
        self.normalize = normalize
        if data_avail_tolerances is None:
            data_avail_tolerances = [0 for m in machines]
        self.data_avail_tolerances = data_avail_tolerances
        self.is_strictly_positive = is_strictly_positive
        self.mapping_paths = mapping_paths

    def is_strictly_positive_fn(self):
        return self.is_strictly_positive

    def is_ip(self):
        return self.is_ip

    def get_file_path(self, prepath, machine, shot_number):
        signal_dirname = self.get_path(machine)
        dirname = os.path.join(prepath, machine.name, signal_dirname)
        return get_individual_shot_file(dirname, machine.name, shot_number,
                                        raw_signal=True)

    def is_valid(self, prepath, shot, dtype='float32'):
        t, data, exists = self.load_data(prepath, shot, dtype)
        return exists

    def is_saved(self, prepath, shot):
        file_path = self.get_file_path(prepath, shot.machine, shot.number)
        return os.path.isfile(file_path)

    def load_data_from_txt_safe(self, prepath, shot, dtype='float32'):
        file_path = self.get_file_path(prepath, shot.machine, shot.number)
        if not self.is_saved(prepath, shot):
            print('Signal {}, shot {} was never downloaded [omit]'.format(
                self.description, shot.number))
            return None, False

        if os.path.getsize(file_path) == 0:
            print('Signal {}, shot {} '.format(self.description, shot.number),
                  'was downloaded incorrectly (empty file) [omit]')
            os.remove(file_path)
            return None, False
        try:
            data = np.loadtxt(file_path, dtype=dtype)
            if np.all(data == get_missing_value_array()):
                print('Signal {}, shot {} contains no data [omit]'.format(
                    self.description, shot.number))
                return None, False
        except Exception as e:
            print(e)
            print('Cannot load signal {} shot {} [omit]'.format(
                file_path, shot.number))
            os.remove(file_path)
            return None, False

        return data, True

    def load_data(self, prepath, shot, dtype='float32'):
        data, succ = self.load_data_from_txt_safe(prepath, shot)
        if not succ:
            return None, None, False

        if np.ndim(data) == 1:
            data = np.expand_dims(data, axis=0)

        t = data[:, 0]
        sig = data[:, 1:]

        if self.is_ip:  # restrict shot to current threshold
            region = np.where(np.abs(sig) >= shot.machine.current_threshold)[0]
            if len(region) == 0:
                print('Shot {} has no current [omit]'.format(shot.number))
                return None, sig.shape, False
            first_idx = region[0]
            last_idx = region[-1]
            # add 50 ms to cover possible disruption event
            last_time = t[last_idx] + 5e-2
            last_indices = np.where(t > last_time)[0]
            if len(last_indices) == 0:
                last_idx = -1
            else:
                last_idx = last_indices[0]
            t = t[first_idx:last_idx]
            sig = sig[first_idx:last_idx, :]

        # make sure shot is not garbage data
        if len(t) <= 1 or (np.max(sig) == 0.0 and np.min(sig) == 0.0):
            if self.is_ip:
                print('Shot {} has no current [omit]'.format(shot.number))
            else:
                print('Signal {}, shot {} contains no data [omit]'.format(
                    self.description, shot.number))
            return None, sig.shape, False

        # make sure data doesn't contain NaN values
        if np.any(np.isnan(t)) or np.any(np.isnan(sig)):
            print('Signal {}, shot {} contains NaN [omit]'.format(
                self.description, shot.number))
            return None, sig.shape, False

        return t, sig, True

    def fetch_data_basic(self, machine, shot_num, c, path=None):
        if path is None:
            path = self.get_path(machine)
        success = False
        mapping = None
        try:
            time, data, mapping, success = machine.fetch_data_fn(
                path, shot_num, c)
        except Exception as e:
            print(e)
            sys.stdout.flush()

        if not success:
            return None, None, None, False

        time = np.array(time) + 1e-3*self.get_causal_shift(machine)
        return time, np.array(data), mapping, success

    def fetch_data(self, machine, shot_num, c):
        return self.fetch_data_basic(machine, shot_num, c)

    def is_defined_on_machine(self, machine):
        return machine in self.machines

    def is_defined_on_machines(self, machines):
        return all([m in self.machines for m in machines])

    def get_path(self, machine):
        idx = self.get_idx(machine)
        return self.paths[idx]

    def get_mapping_path(self, machine):
        if self.mapping_paths is None:
            return None
        else:
            idx = self.get_idx(machine)
            return self.mapping_paths[idx]

    def get_causal_shift(self, machine):
        idx = self.get_idx(machine)
        return self.causal_shifts[idx]

    def get_data_avail_tolerance(self, machine):
        idx = self.get_idx(machine)
        return self.data_avail_tolerances[idx]

    def get_idx(self, machine):
        assert machine in self.machines
        idx = self.machines.index(machine)
        return idx

    def description_plus_paths(self):
        return self.description + ' ' + ' '.join(self.paths)

    def __eq__(self, other):
        if other is None:
            return False
        return self.description_plus_paths().__eq__(
            other.description_plus_paths())

    def __ne__(self, other):
        return self.description_plus_paths().__ne__(
            other.description_plus_paths())

    def __lt__(self, other):
        return self.description_plus_paths().__lt__(
            other.description_plus_paths())

    def __hash__(self):
        return myhash(self.description_plus_paths())

    def __str__(self):
        return self.description

    def __repr__(self):
        return self.description


###############################################################################
# Profile (1D) Signal Class
###############################################################################
class ProfileSignal(Signal):
    def __init__(self, description, paths, machines, tex_label=None,
                 causal_shifts=None, mapping_range=(0, 1), num_channels=32,
                 data_avail_tolerances=None, is_strictly_positive=False,
                 mapping_paths=None):
        super(ProfileSignal, self).__init__(
            description, paths, machines, tex_label, causal_shifts,
            is_ip=False, data_avail_tolerances=data_avail_tolerances,
            is_strictly_positive=is_strictly_positive,
            mapping_paths=mapping_paths)
        self.mapping_range = mapping_range
        self.num_channels = num_channels

    def load_data(self, prepath, shot, dtype='float32'):
        data, succ = self.load_data_from_txt_safe(prepath, shot)
        if not succ:
            return None, None, False

        if np.ndim(data) == 1:
            data = np.expand_dims(data, axis=0)
        # time is stored twice, once for mapping and once for signal
        T = data.shape[0]//2
        mapping = data[:T, 1:]
        remapping = np.linspace(self.mapping_range[0], self.mapping_range[1],
                                self.num_channels)
        t = data[:T, 0]
        sig = data[T:, 1:]
        if sig.shape[1] < 2:
            print('Signal {}, shot {} '.format(self.description, shot.number),
                  'should be profile but has only one channel. Possibly only ',
                  'one profile fit was run for the duration of the shot and ',
                  'was transposed during downloading. Need at least 2 channels'
                  ' [omit]')
            return None, None, False
        if len(t) <= 1 or (np.max(sig) == 0.0 and np.min(sig) == 0.0):
            print('Signal {}, shot {} '.format(self.description, shot.number),
                  'contains no data [omit]')
            return None, None, False
        if np.any(np.isnan(t)) or np.any(np.isnan(sig)):
            print('Signal {}, shot {} '.format(self.description, shot.number),
                  'contains NaN value(s) [omit]')
            return None, None, False

        timesteps = len(t)
        sig_interp = np.zeros((timesteps, self.num_channels))
        for i in range(timesteps):
            # make sure the mapping is ordered and unique
            _, order = np.unique(mapping[i, :], return_index=True)
            if sig[i, order].shape[0] > 2:
                # ext = 0 is extrapolation, ext = 3 is boundary value.
                f = UnivariateSpline(mapping[i, order], sig[i, order], s=0,
                                     k=1, ext=3)
                sig_interp[i, :] = f(remapping)
            else:
                print('Signal {}, shot {} '.format(self.description,
                                                   shot.number),
                      'has insufficient points for linear interpolation. ',
                      'dfitpack.error: (m>k) failed for hidden m: fpcurf0:m=1 '
                      '[omit]')
                return None, None, False

        return t, sig_interp, True

    def fetch_data(self, machine, shot_num, c):
        time, data, mapping, success = self.fetch_data_basic(
            machine, shot_num, c)
        path = self.get_path(machine)
        mapping_path = self.get_mapping_path(machine)

        if mapping is not None and np.ndim(mapping) == 1:
            # make sure there is a mapping for every timestep
            T = len(time)
            mapping = np.tile(mapping, (T, 1)).transpose()
            assert mapping.shape == data.shape, (
                "mapping and data shapes are different")
        if mapping_path is not None:
            # fetch the mapping separately
            (time_map, data_map, mapping_map,
             success_map) = self.fetch_data_basic(machine, shot_num, c,
                                                  path=mapping_path)
            success = (success and success_map)
            if not success:
                print("No success for signal {} and mapping {}".format(
                    path, mapping_path))
            else:
                assert np.all(time == time_map), (
                    "time for signal {} and mapping {} ".format(path,
                                                                mapping_path)
                    + "don't align: \n{}\n\n{}\n".format(time, time_map))
                mapping = data_map

        if not success:
            return None, None, None, False
        return time, data, mapping, success


###############################################################################
# Channel Signal Class
###############################################################################
class ChannelSignal(Signal):
    def __init__(self, description, paths, machines, tex_label=None,
                 causal_shifts=None, data_avail_tolerances=None,
                 is_strictly_positive=False, mapping_paths=None):
        super(ChannelSignal, self).__init__(
            description, paths, machines, tex_label, causal_shifts,
            is_ip=False, data_avail_tolerances=data_avail_tolerances,
            is_strictly_positive=is_strictly_positive,
            mapping_paths=mapping_paths)
        nums, new_paths = self.get_channel_nums(paths)
        self.channel_nums = nums
        self.paths = new_paths

    def get_channel_nums(self, paths):
        regex = re.compile(r'channel\d+')
        regex_int = re.compile(r'\d+')
        nums = []
        new_paths = []
        for p in paths:
            assert p[-1] != '/'
            elements = p.split('/')
            res = regex.findall(elements[-1])
            assert len(res) < 2
            if len(res) == 0:
                nums.append(None)
                new_paths.append(p)
            else:
                nums.append(int(regex_int.findall(res[0])[0]))
                new_paths.append("/".join(elements[:-1]))
        return nums, new_paths

    def get_channel_num(self, machine):
        idx = self.get_idx(machine)
        return self.channel_nums[idx]

    def fetch_data(self, machine, shot_num, c):
        time, data, mapping, success = self.fetch_data_basic(
            machine, shot_num, c)
        mapping = None  # we are not interested in the whole profile
        channel_num = self.get_channel_num(machine)
        if channel_num is not None and success:
            if np.ndim(data) != 2:
                print("Channel Signal {} expected 2D array for shot {}".format(
                    self, self.shot_number), ' [omit]')
                success = False
            else:
                data = data[channel_num, :]  # extract channel of interest
        return time, data, mapping, success

    def get_file_path(self, prepath, machine, shot_number):
        signal_dirname = self.get_path(machine)
        num = self.get_channel_num(machine)
        if num is not None:
            # TODO(KGF): deduplicate with parent class fn. Only difference:
            signal_dirname += "/channel{}".format(num)
        dirname = os.path.join(prepath, machine.name, signal_dirname)
        return get_individual_shot_file(dirname, machine.name, shot_number,
                                        raw_signal=True)


###############################################################################
# 2-Dimensional Signal Class
###############################################################################
class Signal2D(Signal):
    """
    Signal2D is a signal class specifically tailored for two-dimensional
    signals, such as ECEi data

    Non-inherited Attributes:
        dims: tuple of ints, dimensions of 2d signal; ((20, 8) for ECEi)
        is_ecei: bool, True if data is ECEi data
        miss_chan_threshold: int, number of channels that can be
                             missing in order for a shot to be included
    """
    def __init__(self, description, paths, machines, dims, is_ecei = False, 
                 miss_chan_threshold = 80, tex_label=None, causal_shifts=None,
                 is_ip=False, normalize=True, data_avail_tolerances=None,
                 is_strictly_positive=False, mapping_paths=None):
        super(Signal2D, self).__init__(
            description, paths, machines,
            tex_label=tex_label, causal_shifts=causal_shifts,
            is_ip=False, normalize=normalize,
            data_avail_tolerances=data_avail_tolerances,
            is_strictly_positive=is_strictly_positive,
            mapping_paths=mapping_paths)
        self.dims = dims
        self.num_channels = dims[0]*dims[1]
        self.is_ecei = is_ecei


    def get_file_path(self, prepath, machine, shot_number):
        """
        Returns file path.

        Args:
            prepath: str, file prepath
            machine: Machine object, machine that signal is defined on
            shot_number: int, shot number
        """
        if self.is_ecei:
            return prepath+'/'+str(shot_number)+'.hdf5'
        signal_dirname = self.get_path(machine)
        dirname = os.path.join(prepath, machine.name, signal_dirname)
        return get_individual_shot_file(dirname, machine.name, shot_number,
                                        raw_signal=True)


    def load_data_from_hdf5_safe(self, prepath, shot):
        """
        Loads 2D data from hdf5 file where each database in the file contains
        data from a single channel. Stacks data into 2D numpy array with shape
        (time_steps, num_channels+1) (where time series is first column) and 
        returns it. Pads missing channel data with 0's up to the missing channel
        threshold.

        Args:
            prepath: str, location of data
            shot: Shot object for shot number of interest
        """
        file_path = self.get_file_path(prepath, shot.machine, shot.number)
        if not self.is_saved(prepath, shot):
            print('Signal {}, shot {} was never downloaded [omit]'.format(
                self.description, shot.number))
            return None, False

        if os.path.getsize(file_path) == 0:
            print('Signal {}, shot {} '.format(self.description, shot.number),
                  'was downloaded incorrectly (empty file) [omit]')
            os.remove(file_path)
            return None, False

        if self.is_ecei:
            try:
                E = ECEI()
                f = h5py.File(file_path, 'r')
                miss_count = 0
                missing = []
                for key in f.keys():
                    if key.startswith('missing'):
                        miss_count += 1
                        missing.append(key)
                if miss_count == 160:
                    print('Signal {}, shot {} contains no data [omit]'.format(
                          self.description, shot.number))
                    return None, False
                if miss_count > self.miss_chan_threshold:
                    print('Signal {}, shot {} is missing too many channels \
                           [omit]'.format(self.description, shot.number))
                    return None, False

                no_time_series = True
                idx = 0
                while no_time_series:
                    chan = E.ecei_channels[idx]
                    if chan not in missing:
                        data = np.asarray(f.get(chan))[:,0]
                        data = data.reshape((data.shape[0],1))
                        no_time_series = False
                    idx += 1

                for channel in E.ecei_channels:
                    if channel in missing:
                        chan = np.zeros((data.shape[0],1))
                        data = np.append(data, chan, axis = 1)
                    else:
                        chan = np.asarray(f.get(channel))
                        data = np.append(data, chan[:,1].reshape((chan.shape[0],1)),\
                                     axis = 1)
            except Exception as e:
                print(e)
                print('Cannot load signal {} shot {} [omit]'.format(
                      file_path, shot.number))
                os.remove(file_path)
                return None, False
            assert data.shape[1] == 161

        else:
            print('Non-ECEi 2D hdf5 data not yet supported.')
            return None, False

        return data, True


    def load_data_from_txt_safe(self, prepath, shot, dtype='float32'):
        file_path = self.get_file_path(prepath, shot.machine, shot.number)
        if not self.is_saved(prepath, shot):
            print('Signal {}, shot {} was never downloaded [omit]'.format(
                self.description, shot.number))
            return None, False

        if os.path.getsize(file_path) == 0:
            print('Signal {}, shot {} '.format(self.description, shot.number),
                  'was downloaded incorrectly (empty file) [omit]')
            os.remove(file_path)
            return None, False
        try:
            data = np.loadtxt(file_path, dtype=dtype)
            if np.all(data == get_missing_value_array()):
                print('Signal {}, shot {} contains no data [omit]'.format(
                    self.description, shot.number))
                return None, False
        except Exception as e:
            print(e)
            print('Cannot load signal {} shot {} [omit]'.format(
                file_path, shot.number))
            os.remove(file_path)
            return None, False

        return data, True

    def load_data(self, prepath, shot, dtype='float32'):
        if self.is_ecei:
            data, succ = self.load_data_from_hdf5_safe(prepath, shot)
        else:
            data, succ = self.load_data_from_txt_safe(prepath, shot)

        if not succ:
            return None, None, False

        if np.ndim(data) == 1:
            data = np.expand_dims(data, axis=0)

        t = data[:, 0]
        sig = data[:, 1:]

        # make sure shot is not garbage data
        if len(t) <= 1 or (np.max(sig) == 0.0 and np.min(sig) == 0.0):
            if self.is_ip:
                print('Shot {} has no current [omit]'.format(shot.number))
            else:
                print('Signal {}, shot {} contains no data [omit]'.format(
                    self.description, shot.number))
            return None, sig.shape, False

        # make sure data doesn't contain NaN values
        if np.any(np.isnan(t)) or np.any(np.isnan(sig)):
            print('Signal {}, shot {} contains NaN [omit]'.format(
                self.description, shot.number))
            return None, sig.shape, False

        return t, sig, True

    def fetch_data_basic(self, machine, shot_num, c, path=None):
        success = False
        if self.is_ecei:
            E = ECEI()
            time, data, mapping, success = E.Fetch_Shot(shot_num)
        else:
            if path is None:
                path = self.get_path(machine)
            mapping = None
            try:
                time, data, mapping, success = machine.fetch_data_fn(
                    path, shot_num, c)
            except Exception as e:
                print(e)
                sys.stdout.flush()

        if not success:
            return None, None, None, False

        time = np.array(time) + 1e-3*self.get_causal_shift(machine)
        return time, np.array(data), mapping, success

    def fetch_data(self, machine, shot_num, c):
        return self.fetch_data_basic(machine, shot_num, c)


class Machine(object):
    def __init__(self, name, server, fetch_data_fn, max_cores=8,
                 current_threshold=0):
        self.name = name
        self.server = server
        self.max_cores = max_cores
        self.fetch_data_fn = fetch_data_fn
        self.current_threshold = current_threshold

    def get_connection(self):
        return Connection(self.server)

    def __eq__(self, other):
        return self.name.__eq__(other.name)

    def __lt__(self, other):
        return self.name.__lt__(other.name)

    def __ne__(self, other):
        return self.name.__ne__(other.name)

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
