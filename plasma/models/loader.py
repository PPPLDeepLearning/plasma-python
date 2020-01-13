'''
#########################################################
This file containts classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function, division
import numpy as np

from plasma.primitives.shots import Shot
import multiprocessing as mp

# import pdb


class Loader(object):
    '''A Python class to ...

    The length of shots in e.g. JET data varies by orders of magnitude. For
    data parallel synchronous training it is essential that amounds of train
    data passed to the model replica is about the same size.  Therefore, a
    patching technique is introduced.

    A patch is a subset of shot's time/signal profile having a fixed length,
    equal among all patches.  Patch size is approximately equal to the minimum
    shot length. More precisely: it is equal to the max(1,
    min_len//rnn_length)*rnn_length - the largest number less or equal to the
    minimum shot length divisible by the LSTM model length. If minimum shot
    length is less than the rnn_length, then the patch length is equal to the
    rnn_length

    '''

    def __init__(self, conf, normalizer=None):
        self.conf = conf
        self.stateful = conf['model']['stateful']
        self.normalizer = normalizer
        self.verbose = True

    def set_inference_mode(self, val):
        self.normalizer.set_inference_mode(val)

    def training_batch_generator(self, shot_list):
        """The method implements a training batch generator as a Python
        generator with a while-loop.  It iterates indefinitely over the
        data set and returns one mini-batch of data at a time.

        NOTE: Can be inefficient during distributed training because one
        process loading data will cause all other processes to stall.

        Argument list:
          - shot_list:

        Returns:
          - One mini-batch of data and label as a Numpy array: X[start:end],
        y[start:end]
          - reset_states_now: boolean flag indicating when to reset state
        during stateful RNN training
          - num_so_far,num_total: number of samples generated so far and the
        total dataset size as per shot_list

        """
        batch_size = self.conf['training']['batch_size']
        num_at_once = self.conf['training']['num_shots_at_once']
        epoch = 0
        num_so_far = 0
        while True:
            # the list of all shots
            shot_list.shuffle()
            # split the list into equal-length sublists (random shots will be
            # reused to make them equal length).
            shot_sublists = shot_list.sublists(num_at_once, equal_size=True)
            num_total = len(shot_list)
            for (i, shot_sublist) in enumerate(shot_sublists):
                # produce a list of equal-length chunks from this set of shots
                X_list, y_list = self.load_as_X_y_list(shot_sublist)
                # Each chunk will be a multiple of the batch size
                for j, (X, y) in enumerate(zip(X_list, y_list)):
                    num_examples = X.shape[0]
                    assert num_examples % batch_size == 0
                    num_chunks = num_examples//batch_size
                    """
                    The method produces batch-sized training data X and labels
                    y as Numpy arrays to feed during training.

                    Mini-batch dimensions are (num_examples, num_timesteps,
                    num_dimensions_of_data) also num_examples has to be
                    divisible by the batch_size. The i-th example and the
                    (batchsize + 1)-th example are consecutive in time, so we
                    do not reset the RNN internal state unless we start a new
                    chunk.

                    """
                    for k in range(num_chunks):
                        # epoch_end = (i == len(shot_sublists) - 1 and j ==
                        # len(X_list) -1 and k == num_chunks - 1)
                        reset_states_now = (k == 0)
                        start = k*batch_size
                        end = (k + 1)*batch_size
                        num_so_far += 1.0 * \
                            len(shot_sublist)/(len(X_list)*num_chunks)
                        yield X[start:end], y[start:end], reset_states_now,
                        num_so_far, num_total
            epoch += 1

    def fill_training_buffer(self, Xbuff, Ybuff, end_indices, shot,
                             is_first_fill=False):
        sig, res = self.get_signal_result_from_shot(shot)
        length = self.conf['model']['length']
        if is_first_fill:  # cut signal to random position
            cut_idx = np.random.randint(res.shape[0]-length+1)
            sig = sig[cut_idx:]
            res = res[cut_idx:]

        sig_len = res.shape[0]
        sig_len = (sig_len // length)*length  # make divisible by lenth
        assert sig_len > 0
        batch_idx = np.where(end_indices == 0)[0][0]
        if sig_len > Xbuff.shape[1]:
            Xbuff = self.resize_buffer(Xbuff, sig_len+length)
            Ybuff = self.resize_buffer(Ybuff, sig_len+length)
        Xbuff[batch_idx, :sig_len, :] = sig[-sig_len:]
        Ybuff[batch_idx, :sig_len, :] = res[-sig_len:]
        end_indices[batch_idx] += sig_len
        # print("Filling buffer at index {}".format(batch_idx))
        return Xbuff, Ybuff, batch_idx

    def return_from_training_buffer(self, Xbuff, Ybuff, end_indices):
        length = self.conf['model']['length']
        end_indices -= length
        assert np.all(end_indices >= 0)
        X = 1.0*Xbuff[:, :length, :]
        Y = 1.0*Ybuff[:, :length, :]
        self.shift_buffer(Xbuff, length)
        self.shift_buffer(Ybuff, length)
        return X, Y

    def shift_buffer(self, buff, length):
        buff[:, :-length, :] = buff[:, length:, :]

    def resize_buffer(self, buff, new_length, dtype=None):
        if dtype is None:
            dtype = self.conf['data']['floatx']
        old_length = buff.shape[1]
        batch_size = buff.shape[0]
        num_signals = buff.shape[2]
        new_buff = np.zeros((batch_size, new_length, num_signals), dtype=dtype)
        new_buff[:, :old_length, :] = buff
        # print("Resizing buffer to new length {}".format(new_length))
        return new_buff

    def inference_batch_generator_full_shot(self, shot_list):
        """
        The method implements a training batch generator as a Python generator
        with a while-loop.

        It iterates indefinitely over the data set and returns one mini-batch
        of data at a time.


        NOTE: Can be inefficient during distributed training because one
        process loading data will cause all other processes to stall.

        Argument list:
          - shot_list:

        Returns:
          - One mini-batch of data and label as a Numpy array:
                 X[start:end],y[start:end]
          - reset_states_now: boolean flag indicating when to reset state
            during stateful RNN training
          - num_so_far,num_total: number of samples generated so far and
            the total dataset size as per shot_list
        """
        batch_size = self.conf['model']['pred_batch_size']
        sig, res = self.get_signal_result_from_shot(shot_list.shots[0])
        Xbuff = np.zeros((batch_size,) + sig.shape,
                         dtype=self.conf['data']['floatx'])
        Ybuff = np.zeros((batch_size,) + res.shape,
                         dtype=self.conf['data']['floatx'])
        Maskbuff = np.zeros((batch_size,) + res.shape,
                            dtype=self.conf['data']['floatx'])
        disr = np.zeros(batch_size, dtype=bool)
        lengths = np.zeros(batch_size, dtype=int)
        # epoch = 0
        num_total = len(shot_list)
        num_so_far = 0
        #  returned = False
        #  num_steps = 0
        batch_idx = 0
        np.seterr(all='raise')
        # warmup_steps = self.conf['training']['batch_generator_warmup_steps']
        # is_warmup_period = num_steps < warmup_steps
        # is_first_fill = num_steps < batch_size
        while True:
            # the list of all shots
            # shot_list.shuffle()
            for i in range(num_total):
                shot = shot_list.shots[i]
                sig, res = self.get_signal_result_from_shot(shot)
                sig_len = res.shape[0]
                if sig_len > Xbuff.shape[1]:  # resize buffer if needed
                    old_len = Xbuff.shape[1]
                    Xbuff = self.resize_buffer(Xbuff, sig_len)
                    Ybuff = self.resize_buffer(Ybuff, sig_len)
                    Maskbuff = self.resize_buffer(Maskbuff, sig_len)
                    Maskbuff[:, old_len:, :] = 0.0

                Xbuff[batch_idx, :, :] = 0.0
                Ybuff[batch_idx, :, :] = 0.0
                Maskbuff[batch_idx, :, :] = 0.0
                Xbuff[batch_idx, :sig_len, :] = sig
                Ybuff[batch_idx, :sig_len, :] = res
                Maskbuff[batch_idx, :sig_len, :] = 1.0
                disr[batch_idx] = shot.is_disruptive_shot()
                lengths[batch_idx] = res.shape[0]
                batch_idx += 1
                if batch_idx == batch_size:
                    num_so_far += batch_size
                    x1 = 1.0*Xbuff
                    x2 = 1.0*Ybuff
                    x3 = 1.0*Maskbuff
                    x4 = disr & True
                    x5 = 1*lengths

                    yield x1, x2, x3, x4, x5, num_so_far, num_total
                    batch_idx = 0

    def training_batch_generator_full_shot_partial_reset(self, shot_list):
        """

        The method implements a training batch generator as a Python generator
        with a while-loop. It iterates indefinitely over the data set and
        returns one mini-batch of data at a time.

        NOTE: Can be inefficient during distributed training because one
        process loading data will cause all other processes to stall.

        Argument list:
          - shot_list:

        Returns:
          - One mini-batch of data and label as a Numpy array:
            X[start:end],y[start:end]
          - reset_states_now: boolean flag indicating when to reset state
            during stateful RNN training
          - num_so_far,num_total: number of samples generated so far and the
            total dataset size as per shot_list
        """
        batch_size = self.conf['training']['batch_size']
        sig, res = self.get_signal_result_from_shot(shot_list.shots[0])
        Xbuff = np.empty((batch_size,) + sig.shape,
                         dtype=self.conf['data']['floatx'])
        Ybuff = np.empty((batch_size,) + res.shape,
                         dtype=self.conf['data']['floatx'])
        Maskbuff = np.empty((batch_size,) + res.shape,
                            dtype=self.conf['data']['floatx'])
        # epoch = 0
        num_total = len(shot_list)
        num_so_far = 0
        batch_idx = 0
        # warmup_steps = self.conf['training']['batch_generator_warmup_steps']
        # is_warmup_period = num_steps < warmup_steps
        # is_first_fill = num_steps < batch_size
        while True:
            # the list of all shots
            shot_list.shuffle()
            for i in range(num_total):
                shot = self.sample_shot_from_list_given_index(shot_list, i)
                sig, res = self.get_signal_result_from_shot(shot)
                sig_len = res.shape[0]
                if sig_len > Xbuff.shape[1]:  # resize buffer if needed
                    old_len = Xbuff.shape[1]
                    Xbuff = self.resize_buffer(Xbuff, sig_len)
                    Ybuff = self.resize_buffer(Ybuff, sig_len)
                    Maskbuff = self.resize_buffer(Maskbuff, sig_len)
                    Maskbuff[:, old_len:, :] = 0.0

                Xbuff[batch_idx, :, :] = 0.0
                Ybuff[batch_idx, :, :] = 0.0
                Maskbuff[batch_idx, :, :] = 0.0

                Xbuff[batch_idx, :sig_len, :] = sig
                Ybuff[batch_idx, :sig_len, :] = res
                Maskbuff[batch_idx, :sig_len, :] = 1.0
                batch_idx += 1
                if batch_idx == batch_size:
                    num_so_far += batch_size
                    yield (1.0*Xbuff, 1.0*Ybuff, 1.0*Maskbuff, num_so_far,
                           num_total)
                    batch_idx = 0

    def sample_shot_from_list_given_index(self, shot_list, i):
        if self.conf['training']['ranking_difficulty_fac'] == 1.0:
            if self.conf['data']['equalize_classes']:
                shot = shot_list.sample_equal_classes()
            else:
                shot = shot_list.shots[i]
        else:  # draw the shot weighted
            shot = shot_list.sample_weighted()
        return shot

    def training_batch_generator_partial_reset(self, shot_list):
        """
        The method implements a training batch generator as a Python generator
        with a while-loop. It iterates indefinitely over the data set and
        returns one mini-batch of data at a time.

        NOTE: Can be inefficient during distributed training because one
        process loading data will cause all other processes to stall.

        Argument list:
          - shot_list:

        Returns:
          - One mini-batch of data and label as a Numpy array: X[start:end],
        y[start:end]
          - reset_states_now: boolean flag indicating when to reset state
        during stateful RNN training
          - num_so_far,num_total: number of samples generated so far and the
        total dataset size as per shot_list
        """
        batch_size = self.conf['training']['batch_size']
        # length = self.conf['model']['length']
        sig, res = self.get_signal_result_from_shot(shot_list.shots[0])
        Xbuff = np.empty((batch_size,) + sig.shape,
                         dtype=self.conf['data']['floatx'])
        Ybuff = np.empty((batch_size,) + res.shape,
                         dtype=self.conf['data']['floatx'])
        end_indices = np.zeros(batch_size, dtype=np.int)
        batches_to_reset = np.ones(batch_size, dtype=np.bool)
        # epoch = 0
        num_total = len(shot_list)
        num_so_far = 0
        returned = False
        num_steps = 0
        warmup_steps = self.conf['training']['batch_generator_warmup_steps']
        is_warmup_period = num_steps < warmup_steps
        is_first_fill = num_steps < batch_size
        while True:
            # the list of all shots
            shot_list.shuffle()
            for i in range(len(shot_list)):
                if self.conf['training']['ranking_difficulty_fac'] == 1.0:
                    if self.conf['data']['equalize_classes']:
                        shot = shot_list.sample_equal_classes()
                    else:
                        # TODO(KGF): test my merge of the "jdev" branch for the
                        # following line into this set of conditionals
                        shot = self.sample_shot_from_list_given_index(
                            shot_list, i)
                        # shot = shot_list.shots[i]
                else:  # draw the shot weighted
                    shot = shot_list.sample_weighted()
                while not np.any(end_indices == 0):
                    X, Y = self.return_from_training_buffer(
                        Xbuff, Ybuff, end_indices)
                    yield (X, Y, batches_to_reset, num_so_far, num_total,
                           is_warmup_period)
                    returned = True
                    num_steps += 1
                    is_warmup_period = num_steps < warmup_steps
                    is_first_fill = num_steps < batch_size
                    batches_to_reset[:] = False

                Xbuff, Ybuff, batch_idx = self.fill_training_buffer(
                    Xbuff, Ybuff, end_indices, shot, is_first_fill)
                batches_to_reset[batch_idx] = True
                if returned and not is_warmup_period:
                    num_so_far += 1
            # epoch += 1

    def fill_batch_queue(self, shot_list, queue):
        print("Starting thread to fill queue")
        gen = self.training_batch_generator_partial_reset(shot_list)
        while True:
            ret = next(gen)
            queue.put(ret, block=True, timeout=-1)

    def training_batch_generator_process(self, shot_list):
        queue = mp.Queue()
        proc = mp.Process(target=self.fill_batch_queue,
                          args=(shot_list, queue))
        proc.start()
        while True:
            yield queue.get(True)
        proc.join()
        queue.close()

    def load_as_X_y_list(self, shot_list, prediction_mode=False):
        """
        The method turns a ShotList into a set of equal-sized patches which
        contain a number of examples that is a multiple of the batch size.
        Initially, shots are "light" meaning signal amd disruption related
        attributes are not filled.
        By invoking Loader.get_signals_results_from_shotlist the
        shot information is filled and stored in the object in memory. Next,
        patches are made, finally patches are arranged into batch input shape
        expected by RNN model.

        Performs calls to: get_signals_results_from_shotlist, make_patches,
        arange_patches

        Argument list:
          - shot_list: a ShotList
          - prediction_mode: unused

        Returns:
          - X_list,y_list: lists of Numpy arrays of batch input shape

        """
        # TODO(KGF): check tuple unpack
        (signals, results,
         total_length) = self.get_signals_results_from_shotlist(shot_list)
        sig_patches, res_patches = self.make_patches(signals, results)

        X_list, y_list = self.arange_patches(sig_patches, res_patches)

        effective_length = len(res_patches)*len(res_patches[0])
        if self.verbose:
            print('multiplication factor: {}'.format(
                1.0 * effective_length / total_length))
            print('effective/total length : {}/{}'.format(
                effective_length, total_length))
            print('patch length: {} num patches: {}'.format(
                len(res_patches[0]), len(res_patches)))
        return X_list, y_list

    def load_as_X_y_pred(self, shot_list, custom_batch_size=None):
        (signals, results, shot_lengths,
         disruptive) = self.get_signals_results_from_shotlist(
             shot_list, prediction_mode=True)
        sig_patches, res_patches = self.make_prediction_patches(signals,
                                                                results)
        X, y = self.arange_patches_single(sig_patches, res_patches,
                                          prediction_mode=True,
                                          custom_batch_size=custom_batch_size)
        return X, y, shot_lengths, disruptive

    def get_signals_results_from_shotlist(self, shot_list,
                                          prediction_mode=False):
        prepath = self.conf['paths']['processed_prepath']
        use_signals = self.conf['paths']['use_signals']
        signals = []
        results = []
        disruptive = []
        shot_lengths = []
        total_length = 0
        for shot in shot_list:
            assert isinstance(shot, Shot)
            assert shot.valid
            shot.restore(prepath)

            if self.normalizer is not None:
                self.normalizer.apply(shot)
            else:
                print('Warning, no normalization. ',
                      'Training data may be poorly conditioned')

            if self.conf['training']['use_mock_data']:
                signal, ttd = self.get_mock_data()
            ttd, signal = shot.get_data_arrays(
                use_signals, self.conf['data']['floatx'])
            if len(ttd) < self.conf['model']['length']:
                print(ttd)
                print(shot)
                print(shot.number)

            total_length += len(ttd)
            signals.append(signal)
            shot_lengths.append(len(ttd))
            disruptive.append(shot.is_disruptive)
            if len(ttd.shape) == 1:
                results.append(np.expand_dims(ttd, axis=1))
            else:
                results.append(ttd)
            shot.make_light()
        if not prediction_mode:
            return signals, results, total_length
        else:
            return signals, results, shot_lengths, disruptive

    def get_signal_result_from_shot(self, shot, prediction_mode=False):
        prepath = self.conf['paths']['processed_prepath']
        use_signals = self.conf['paths']['use_signals']
        assert isinstance(shot, Shot)
        assert shot.valid
        shot.restore(prepath)
        if self.normalizer is not None:
            self.normalizer.apply(shot)
        else:
            print('Warning, no normalization. ',
                  'Training data may be poorly conditioned')

        if self.conf['training']['use_mock_data']:
            signal, ttd = self.get_mock_data()
        ttd, signal = shot.get_data_arrays(
            use_signals, self.conf['data']['floatx'])
        if len(ttd) < self.conf['model']['length']:
            print(ttd)
            print(shot)
            print(shot.number)
            print("Shot must be at least as long as the RNN length.")
            exit(1)

        if len(ttd.shape) == 1:
            ttd = np.expand_dims(ttd, axis=1)
        shot.make_light()
        if not prediction_mode:
            return signal, ttd
        else:
            return signal, ttd, shot.is_disruptive

    def batch_output_to_array(self, output, batch_size=None):
        if batch_size is None:
            batch_size = self.conf['model']['pred_batch_size']
        assert output.shape[0] % batch_size == 0
        num_chunks = output.shape[0] // batch_size
        num_timesteps = output.shape[1]
        feature_size = output.shape[2]

        outs = []
        for patch_idx in range(batch_size):
            out = np.empty((num_chunks*num_timesteps, feature_size))
            for chunk in range(num_chunks):
                out[chunk*num_timesteps:(chunk + 1)*num_timesteps, :] = output[
                    chunk * batch_size + patch_idx, :, :]
            outs.append(out)
        return outs

    def make_deterministic_patches(self, signals, results):
        num_timesteps = self.conf['model']['length']
        sig_patches = []
        res_patches = []
        min_len = self.get_min_len(signals, num_timesteps)
        for sig, res in zip(signals, results):
            (sig_patch,
             res_patch) = self.make_deterministic_patches_from_single_array(
                sig, res, min_len)
            sig_patches += sig_patch
            res_patches += res_patch
        return sig_patches, res_patches

    def make_deterministic_patches_from_single_array(self, sig, res, min_len):
        sig_patches = []
        res_patches = []
        if len(sig) <= min_len:
            print('signal length: {}'.format(len(sig)))
        assert min_len <= len(sig)
        for start in range(0, len(sig)-min_len, min_len):
            sig_patches.append(sig[start:start+min_len])
            res_patches.append(res[start:start+min_len])
        sig_patches.append(sig[-min_len:])
        res_patches.append(res[-min_len:])
        return sig_patches, res_patches

    def make_random_patches(self, signals, results, num):
        num_timesteps = self.conf['model']['length']
        sig_patches = []
        res_patches = []
        min_len = self.get_min_len(signals, num_timesteps)
        for i in range(num):
            idx = np.random.randint(len(signals))
            sig_patch, res_patch = self.make_random_patch_from_array(
                signals[idx], results[idx], min_len)
            sig_patches.append(sig_patch)
            res_patches.append(res_patch)
        return sig_patches, res_patches

    def make_random_patch_from_array(self, sig, res, min_len):
        start = np.random.randint(len(sig) - min_len+1)
        return sig[start:start+min_len], res[start:start+min_len]

    def get_min_len(self, arrs, length):
        min_len = min([len(a) for a in arrs]
                      + [self.conf['training']['max_patch_length']])
        min_len = max(1, min_len // length) * length
        return min_len

    def get_max_len(self, arrs, length):
        max_len = max([len(a) for a in arrs])
        max_len = int(np.ceil(1.0*max_len / length) * length)
        return max_len

    def make_patches(self, signals, results):
        """A patch is a subset of shot's time/signal profile having a fixed
        length, equal among all patches.  Patch size is approximately equal to
        the minimum shot length. More precisely: it is equal to the max(1,
        min_len//rnn_length)*rnn_length - the largest number less or equal to
        the minimum shot length divisible by the LSTM model length. If minimum
        shot length is less than the rnn_length, then the patch length is equal
        to the rnn_length

        Since shot lengthes are not multiples of the minimum shot length in
        general, some non-deterministic fraction of patches is created. See:

        Deterministic patching:

        Random patching:


        Argument list:
          - signals: a list of 1D Numpy array of doubles containing signal
        values (a plasma property). Numpy arrays are shot-sized
          - results: a list of 1D Numpy array of doubles containing disruption
        times or -1 if a shot is non-disruptive. Numpy arrays are shot-sized

          NOTE: signals and results are parallel lists. Since Arrays are
          shot-sized, the shape veries across the list


        Returns:
          - sig_patches_det + sig_patches_rand: (concatenated) list of 1D Numpy
        arrays of doubles containing signal values. Numpy arrays are
        patch-sized
          - res_patches_det + res_patches_rand: (concatenated) a list of 1D
        Numpy array of doubles containing disruption times or -1 if a shot is
        non-disruptive. Numpy arrays are patch-sized
          NOTE: sig_patches_det + sig_patches_rand and res_patches_det +
        res_patches_rand are prallel lists. All arrays in the list have
        identical shapes.

        """
        total_num = self.conf['training']['batch_size']
        sig_patches_det, res_patches_det = self.make_deterministic_patches(
            signals, results)
        num_already = len(sig_patches_det)

        total_num = int(np.ceil(1.0 * num_already / total_num)) * total_num

        num_additional = total_num - num_already
        assert num_additional >= 0
        sig_patches_rand, res_patches_rand = self.make_random_patches(
            signals, results, num_additional)
        if self.verbose:
            print('random to deterministic ratio: {}/{}'.format(num_additional,
                                                                num_already))
        return (sig_patches_det + sig_patches_rand,
                res_patches_det + res_patches_rand)

    def make_prediction_patches(self, signals, results):
        # total_num = self.conf['training']['batch_size']
        num_timesteps = self.conf['model']['pred_length']
        sig_patches = []
        res_patches = []
        max_len = self.get_max_len(signals, num_timesteps)
        for sig, res in zip(signals, results):
            sig_patches.append(Loader.pad_array_to_length(sig, max_len))
            res_patches.append(Loader.pad_array_to_length(res, max_len))
        return sig_patches, res_patches

    @staticmethod
    def pad_array_to_length(arr, length):
        dlength = max(0, length - arr.shape[0])
        tuples = [(0, dlength)]
        for l in arr.shape[1:]:
            tuples.append((0, 0))
        return np.pad(arr, tuples, mode='constant', constant_values=0)

    def arange_patches(self, sig_patches, res_patches):
        num_timesteps = self.conf['model']['length']
        batch_size = self.conf['training']['batch_size']
        assert len(sig_patches) % batch_size == 0  # fixed number of batches
        # divisible by length of RNN sequence
        assert len(sig_patches[0]) % num_timesteps == 0
        num_batches = len(sig_patches) // batch_size
        # patch_length = len(sig_patches[0])

        zipped = list(zip(sig_patches, res_patches))
        np.random.shuffle(zipped)
        sig_patches, res_patches = zip(*zipped)
        X_list = []
        y_list = []
        for i in range(num_batches):
            X, y = self.arange_patches_single(
                sig_patches[i*batch_size:(i+1)*batch_size],
                res_patches[i*batch_size:(i+1)*batch_size])
            X_list.append(X)
            y_list.append(y)
        return X_list, y_list

    def arange_patches_single(self, sig_patches, res_patches,
                              prediction_mode=False, custom_batch_size=None):
        if prediction_mode:
            num_timesteps = self.conf['model']['pred_length']
            batch_size = self.conf['model']['pred_batch_size']
        else:
            num_timesteps = self.conf['model']['length']
            batch_size = self.conf['training']['batch_size']
        return_sequences = self.conf['model']['return_sequences']
        if custom_batch_size is not None:
            batch_size = custom_batch_size

        assert len(sig_patches) == batch_size
        assert len(sig_patches[0]) % num_timesteps == 0
        num_chunks = len(sig_patches[0]) // num_timesteps
        num_dimensions_of_data = sig_patches[0].shape[1]
        if len(res_patches[0].shape) == 1:
            num_answers = 1
        else:
            num_answers = res_patches[0].shape[1]

        X = np.zeros((num_chunks*batch_size, num_timesteps,
                      num_dimensions_of_data))
        if return_sequences:
            y = np.zeros((num_chunks*batch_size, num_timesteps, num_answers))
        else:
            y = np.zeros((num_chunks*batch_size, num_answers))

        for chunk_idx in range(num_chunks):
            src_start = chunk_idx*num_timesteps
            src_end = (chunk_idx+1)*num_timesteps
            for patch_idx in range(batch_size):
                X[chunk_idx*batch_size + patch_idx, :,
                    :] = sig_patches[patch_idx][src_start:src_end]
                if return_sequences:
                    y[chunk_idx*batch_size + patch_idx, :,
                        :] = res_patches[patch_idx][src_start:src_end]
                else:
                    y[chunk_idx*batch_size + patch_idx,
                        :] = res_patches[patch_idx][src_end-1]
        return X, y

    def load_as_X_y(self, shot, prediction_mode=False):
        assert isinstance(shot, Shot)
        assert shot.valid
        prepath = self.conf['paths']['processed_prepath']
        return_sequences = self.conf['model']['return_sequences']
        shot.restore(prepath)

        if self.normalizer is not None:
            self.normalizer.apply(shot)
        else:
            print('Warning, no normalization. ',
                  'Training data may be poorly conditioned')
        signals = shot.signals
        ttd = shot.ttd

        if self.conf['training']['use_mock_data']:
            signals, ttd = self.get_mock_data()

        # if not self.stateful:
        #     X,y = self.array_to_path_and_external_pred(signals,ttd)
        # else:
        X, y = self.array_to_path_and_external_pred_cut(
            signals, ttd, return_sequences=return_sequences,
            prediction_mode=prediction_mode)

        shot.make_light()
        return X, y  # X,y

    def get_mock_data(self):
        signals = np.linspace(0, 4*np.pi, 10000)
        rand_idx = np.randint(6000)
        lgth = np.randint(1000, 3000)
        signals = signals[rand_idx:rand_idx+lgth]
        # ttd[-100:] = 1
        signals = np.vstack([signals]*8)
        signals = signals.T
        signals[:, 0] = 0.5 + 0.5*np.sin(signals[:, 0])
        signals[:, 1] = 0.5  # + 0.5*cos(signals[:,1])
        signals[:, 2] = 0.5 + 0.5*np.sin(2*signals[:, 2])
        signals[:, 3:] *= 0
        offset = 100
        ttd = 0.0*signals[:, 0]
        ttd[offset:] = 1.0*signals[:-offset, 0]
        mask = ttd > np.mean(ttd)
        ttd[~mask] = 0
        # mean(signals[:,:2],1)
        return signals, ttd

    def array_to_path_and_external_pred_cut(
            self, arr, res, return_sequences=False, prediction_mode=False):
        num_timesteps = self.conf['model']['length']
        skip = self.conf['model']['skip']
        if prediction_mode:
            num_timesteps = self.conf['model']['pred_length']
            if not return_sequences:
                num_timesteps = 1
            skip = num_timesteps  # batchsize = 1!
        assert np.shape(arr)[0] == np.shape(res)[0]
        num_chunks = len(arr) // num_timesteps
        arr = arr[-num_chunks*num_timesteps:]
        res = res[-num_chunks*num_timesteps:]
        assert np.shape(arr)[0] == np.shape(res)[0]
        X = []
        y = []
        i = 0

        chunk_range = range(num_chunks-1)
        i_range = range(1, num_timesteps+1, skip)
        if prediction_mode:
            chunk_range = range(num_chunks)
            i_range = range(1)

        for chunk in chunk_range:
            for i in i_range:
                start = chunk*num_timesteps + i
                assert start + num_timesteps <= len(arr)
                X.append(arr[start:start+num_timesteps, :])
                if return_sequences:
                    y.append(res[start:start+num_timesteps])
                else:
                    y.append(res[start+num_timesteps-1:start+num_timesteps])
        X = np.array(X)
        y = np.array(y)
        if len(np.shape(X)) == 1:
            X = np.expand_dims(X, axis=len(np.shape(X)))
        if return_sequences:
            y = np.expand_dims(y, axis=len(np.shape(y)))
        return X, y

    @staticmethod
    def get_batch_size(batch_size, prediction_mode):
        if prediction_mode:
            return 1
        else:
            return batch_size  # Loader.get_num_skips(length,skip)

    @staticmethod
    def get_num_skips(length, skip):
        return 1 + (length-1)//skip


class ProcessGenerator(object):
    def __init__(self, generator):
        self.generator = generator
        self.proc = mp.Process(target=self.fill_batch_queue)
        self.queue = mp.Queue()
        self.proc.start()

    def fill_batch_queue(self):
        print("Starting process to fetch data")
        count = 0
        while True:
            self.queue.put(next(self.generator), True)
            count += 1

    def __next__(self):
        return self.queue.get(True)

    def next(self):
        return self.__next__()

    def __exit__(self):
        self.proc.terminate()
        self.queue.close()
