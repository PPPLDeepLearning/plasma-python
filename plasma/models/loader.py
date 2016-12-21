'''
#########################################################
This file containts classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
import itertools
import numpy as np

from plasma.primitives.shots import Shot

class Loader(object):
    def __init__(self,conf,normalizer=None):
        self.conf = conf
        self.stateful = conf['model']['stateful']
        self.normalizer = normalizer
        self.verbose = True

    def training_batch_generator(self,shot_list):
        """Iterates indefinitely over the data set and returns one batch of data at a time.
        Can be inefficient during distributed training because one process loading data will
        cause all other processes to stall."""
        batch_size = self.conf['training']['batch_size']
        num_at_once = self.conf['training']['num_shots_at_once']
        epoch = 0
        while True:
            num_so_far = 0
            # the list of all shots
            shot_list.shuffle() 
            # split the list into equal-length sublists (random shots will be reused to make them equal length).
            shot_sublists = shot_list.sublists(num_at_once,equal_size=True)
            num_total = len(shot_list)
            for (i,shot_sublist) in enumerate(shot_sublists):
                #produce a list of equal-length chunks from this set of shots
                X_list,y_list = self.load_as_X_y_list(shot_sublist)
                #Each chunk will be a multiple of the batch size
                for j,(X,y) in enumerate(zip(X_list,y_list)):
                    num_examples = X.shape[0]
                    assert(num_examples % batch_size == 0)
                    num_chunks = num_examples/batch_size
                    """produce batch-sized data X,y to feed during training.
                    dimensions are (num_examples, num_timesteps, num_dimensions_of_data)
                    also num_examples is divisible by the batch_size. The ith example and the
                    (batchsize + 1)th example are consecutive in time, so we do not reset the
                    RNN internal state unless we start a new chunk."""
                    for k in range(num_chunks):
                        #epoch_end = (i == len(shot_sublists) - 1 and j == len(X_list) -1 and k == num_chunks - 1)
                        reset_states_now = (k == 0)
                        start = k*batch_size
                        end = (k + 1)*batch_size
                        yield X[start:end],y[start:end],reset_states_now,num_so_far,num_total
                        num_so_far += 1.0*len(shot_sublist)/(len(X_list)*num_chunks)
            epoch += 1






    def load_as_X_y_list(self,shot_list,verbose=False,prediction_mode=False):
        """Turn a list of shots into a set of equal-sized patches which contain a number of examples
        that is a multiple of the batch size."""
        signals,results,total_length = self.get_signals_results_from_shotlist(shot_list) 
        sig_patches, res_patches = self.make_patches(signals,results)

        X_list,y_list = self.arange_patches(sig_patches,res_patches)

        effective_length = len(res_patches)*len(res_patches[0])
        if self.verbose:
            print('multiplication factor: {}'.format(1.0*effective_length/total_length))
            print('effective/total length : {}/{}'.format(effective_length,total_length))
            print('patch length: {} num patches: {}'.format(len(res_patches[0]),len(res_patches)))
        return X_list,y_list

    def load_as_X_y_pred(self,shot_list,verbose=False,custom_batch_size=None):
        signals,results,shot_lengths,disruptive = self.get_signals_results_from_shotlist(shot_list,prediction_mode=True) 
        sig_patches, res_patches = self.make_prediction_patches(signals,results)
        X,y = self.arange_patches_single(sig_patches,res_patches,prediction_mode=True,custom_batch_size=custom_batch_size)
        return X,y,shot_lengths,disruptive


    def get_signals_results_from_shotlist(self,shot_list,prediction_mode=False):
        prepath = self.conf['paths']['processed_prepath']
        signals = []
        results = []
        disruptive = []
        shot_lengths = []
        total_length = 0
        for shot in shot_list:
            assert(isinstance(shot,Shot))
            assert(shot.valid)
            shot.restore(prepath)

            if self.normalizer is not None:
                self.normalizer.apply(shot)
            else:
                print('Warning, no normalization. Training data may be poorly conditioned')



            if self.conf['training']['use_mock_data']:
                sig,res = self.get_mock_data()
                shot.signals = sig
                shot.ttd = res

            total_length += len(shot.ttd)
            signals.append(shot.signals)
            res = shot.ttd
            shot_lengths.append(len(shot.ttd))
            disruptive.append(shot.is_disruptive)
            if len(res.shape) == 1:
                results.append(np.expand_dims(res,axis=1))
            else:
                results.append(shot.ttd)
            shot.make_light()
        if not prediction_mode:
            return signals,results,total_length
        else:
            return signals,results,shot_lengths,disruptive



    def batch_output_to_array(self,output,batch_size = None):
        if batch_size is None:
            batch_size = self.conf['model']['pred_batch_size']
        assert(output.shape[0] % batch_size == 0)
        num_chunks = output.shape[0] / batch_size
        num_timesteps = output.shape[1]
        feature_size = output.shape[2]

        outs = []
        for patch_idx in range(batch_size):
            out = np.empty((num_chunks*num_timesteps,feature_size))
            for chunk in range(num_chunks):
                out[chunk*num_timesteps:(chunk+1)*num_timesteps,:] = output[chunk*batch_size+patch_idx,:,:]
            outs.append(out)
        return outs 


    def make_deterministic_patches(self,signals,results):
        num_timesteps = self.conf['model']['length']
        sig_patches = []
        res_patches = []
        min_len = self.get_min_len(signals,num_timesteps)
        for sig,res in zip(signals,results):
            sig_patch, res_patch =  self.make_deterministic_patches_from_single_array(sig,res,min_len)
            sig_patches += sig_patch
            res_patches += res_patch
        return sig_patches, res_patches

    def make_deterministic_patches_from_single_array(self,sig,res,min_len):
        sig_patches = []
        res_patches = []
        assert(min_len <= len(sig))
        for start in range(0,len(sig)-min_len,min_len):
            sig_patches.append(sig[start:start+min_len])
            res_patches.append(res[start:start+min_len])
        sig_patches.append(sig[-min_len:])
        res_patches.append(res[-min_len:])
        return sig_patches,res_patches

    def make_random_patches(self,signals,results,num):
        num_timesteps = self.conf['model']['length']
        sig_patches = []
        res_patches = []
        min_len = self.get_min_len(signals,num_timesteps)
        for i in range(num):
            idx= np.random.randint(len(signals))
            sig_patch, res_patch =  self.make_random_patch_from_array(signals[idx],results[idx],min_len)
            sig_patches.append(sig_patch)
            res_patches.append(res_patch)
        return sig_patches,res_patches

    def make_random_patch_from_array(self,sig,res,min_len):
        start = np.random.randint(len(sig) - min_len+1)
        return sig[start:start+min_len],res[start:start+min_len]
        

    def get_min_len(self,arrs,length):
        min_len = min([len(a) for a in arrs] + [self.conf['training']['max_patch_length']])
        min_len = max(1,min_len // length) * length 
        return min_len


    def get_max_len(self,arrs,length):
        max_len = max([len(a) for a in arrs])
        max_len = int(np.ceil(1.0*max_len / length) * length )
        return max_len

    def make_patches(self,signals,results):
        total_num = self.conf['training']['batch_size'] 
        sig_patches_det,res_patches_det = self.make_deterministic_patches(signals,results)
        num_already = len(sig_patches_det)
        
        total_num = int(np.ceil(1.0 * num_already / total_num)) * total_num
        
        
        num_additional = total_num - num_already
        assert(num_additional >= 0)
        sig_patches_rand,res_patches_rand = self.make_random_patches(signals,results,num_additional)
        if self.verbose:
            print('random to deterministic ratio: {}/{}'.format(num_additional,num_already))
        return sig_patches_det + sig_patches_rand,res_patches_det + res_patches_rand 



    def make_prediction_patches(self,signals,results):
        total_num = self.conf['training']['batch_size'] 
        num_timesteps = self.conf['model']['pred_length']
        sig_patches = []
        res_patches = []
        max_len = self.get_max_len(signals,num_timesteps)
        for sig,res in zip(signals,results):
            sig_patches.append(Loader.pad_array_to_length(sig,max_len))
            res_patches.append(Loader.pad_array_to_length(res,max_len))
        return sig_patches, res_patches

    @staticmethod
    def pad_array_to_length(arr,length):
        dlength = max(0,length - arr.shape[0])
        tuples = [(0,dlength)]
        for l in arr.shape[1:]:
            tuples.append((0,0))
        return np.pad(arr,tuples,mode='constant',constant_values=0)




    def arange_patches(self,sig_patches,res_patches):
        num_timesteps = self.conf['model']['length']
        batch_size = self.conf['training']['batch_size']
        assert(len(sig_patches) % batch_size == 0) #fixed number of batches
        assert(len(sig_patches[0]) % num_timesteps == 0) #divisible by length of RNN sequence
        num_batches = len(sig_patches) / batch_size
        patch_length = len(sig_patches[0])

        zipped = zip(sig_patches,res_patches)
        np.random.shuffle(zipped)
        sig_patches, res_patches = zip(*zipped) 
        X_list = []
        y_list = []
        for i in range(num_batches):
            X,y = self.arange_patches_single(sig_patches[i*batch_size:(i+1)*batch_size],
                                        res_patches[i*batch_size:(i+1)*batch_size])
            X_list.append(X)
            y_list.append(y)
        return X_list,y_list

    def arange_patches_single(self,sig_patches,res_patches,prediction_mode=False,custom_batch_size=None):
        if prediction_mode:
            num_timesteps = self.conf['model']['pred_length']
            batch_size = self.conf['model']['pred_batch_size']
        else:
            num_timesteps = self.conf['model']['length']
            batch_size = self.conf['training']['batch_size']
        return_sequences = self.conf['model']['return_sequences']
        if custom_batch_size is not None:
            batch_size = custom_batch_size

        assert(len(sig_patches) == batch_size)
        assert(len(sig_patches[0]) % num_timesteps == 0)
        num_chunks = len(sig_patches[0]) / num_timesteps
        num_dimensions_of_data = sig_patches[0].shape[1]
        if len(res_patches[0].shape) == 1:
            num_answers = 1
        else:
            num_answers = res_patches[0].shape[1]
        
        X = np.zeros((num_chunks*batch_size,num_timesteps,num_dimensions_of_data))
        if return_sequences:
            y = np.zeros((num_chunks*batch_size,num_timesteps,num_answers))
        else:
            y = np.zeros((num_chunks*batch_size,num_answers))

        
        for chunk_idx in range(num_chunks):
            src_start = chunk_idx*num_timesteps
            src_end = (chunk_idx+1)*num_timesteps
            for patch_idx in range(batch_size):
                X[chunk_idx*batch_size + patch_idx,:,:] = sig_patches[patch_idx][src_start:src_end]
                if return_sequences:
                    y[chunk_idx*batch_size + patch_idx,:,:] = res_patches[patch_idx][src_start:src_end]
                else:
                    y[chunk_idx*batch_size + patch_idx,:] = res_patches[patch_idx][src_end-1]
        return X,y

    def load_as_X_y(self,shot,verbose=False,prediction_mode=False):
        assert(isinstance(shot,Shot))
        assert(shot.valid)
        prepath = self.conf['paths']['processed_prepath']
        return_sequences = self.conf['model']['return_sequences']
        shot.restore(prepath)

        if self.normalizer is not None:
            self.normalizer.apply(shot)
        else:
            print('Warning, no normalization. Training data may be poorly conditioned')

        signals = shot.signals
        ttd = shot.ttd

        if self.conf['training']['use_mock_data']:
            signals,ttd = self.get_mock_data()

        # if not self.stateful:
        #     X,y = self.array_to_path_and_external_pred(signals,ttd)
        # else:
        X,y = self.array_to_path_and_external_pred_cut(signals,ttd,
                return_sequences=return_sequences,prediction_mode=prediction_mode)

        shot.make_light()
        return  X,y#X,y

    def get_mock_data(self):
        signals = linspace(0,4*pi,10000)
        rand_idx = randint(6000)
        lgth = randint(1000,3000)
        signals = signals[rand_idx:rand_idx+lgth]
        #ttd[-100:] = 1
        signals = vstack([signals]*8)
        signals = signals.T
        signals[:,0] = 0.5 + 0.5*sin(signals[:,0])
        signals[:,1] = 0.5# + 0.5*cos(signals[:,1])
        signals[:,2] = 0.5 + 0.5*sin(2*signals[:,2])
        signals[:,3:] *= 0
        offset = 100
        ttd = 0.0*signals[:,0]
        ttd[offset:] = 1.0*signals[:-offset,0]
        mask = ttd > mean(ttd)
        ttd[~mask] = 0
        #mean(signals[:,:2],1)
        return signals,ttd

    def array_to_path_and_external_pred_cut(self,arr,res,return_sequences=False,prediction_mode=False):
        num_timesteps = self.conf['model']['length']
        skip = self.conf['model']['skip']
        if prediction_mode:
            num_timesteps = self.conf['model']['pred_length']
            if not return_sequences:
                num_timesteps = 1
            skip = num_timesteps #batchsize = 1!
        assert(shape(arr)[0] == shape(res)[0])
        num_chunks = len(arr) // num_timesteps
        arr = arr[-num_chunks*num_timesteps:]
        res = res[-num_chunks*num_timesteps:]
        assert(shape(arr)[0] == shape(res)[0])
        X = []
        y = []
        i = 0

        chunk_range = range(num_chunks-1)
        i_range = range(1,num_timesteps+1,skip)
        if prediction_mode:
            chunk_range = range(num_chunks)
            i_range = range(1)


        for chunk in chunk_range:
            for i in i_range:
                start = chunk*num_timesteps + i
                assert(start + num_timesteps <= len(arr))
                X.append(arr[start:start+num_timesteps,:])
                if return_sequences:
                    y.append(res[start:start+num_timesteps])
                else:
                    y.append(res[start+num_timesteps-1:start+num_timesteps])
        X = array(X)
        y = array(y)
        if len(shape(X)) == 1:
            X = np.expand_dims(X,axis=len(shape(X)))
        if return_sequences:
            y = np.expand_dims(y,axis=len(shape(y)))
        return X,y

    @staticmethod
    def get_batch_size(batch_size,prediction_mode):
        if prediction_mode:
            return 1
        else:
            return batch_size#Loader.get_num_skips(length,skip)

    @staticmethod
    def get_num_skips(length,skip):
        return 1 + (length-1)//skip

    def load_shotlists(self,conf):
        path = conf['paths']['base_path'] + '/normalization/shot_lists.npz'
        data = np.load(path)
        shot_list_train = data['shot_list_train'][()]
        shot_list_validate = data['shot_list_validate'][()]
        shot_list_test = data['shot_list_test'][()]
        return shot_list_train,shot_list_validate,shot_list_test

    # def produce_indices(signals_list):
    #     indices_list = []
    #     for xx in signals_list:
    #         indices_list.append(arange(len(xx)))
    #     return indices_list



    # def array_to_path_and_external_pred(self,arr,res,return_sequences=False):
    #     length = self.conf['model']['length']
    #     skip = self.conf['model']['skip']
    #     assert(shape(arr)[0] == shape(res)[0])
    #     X = []
    #     y = []
    #     i = 0
    #     while True:
    #         pred = i+length
    #         if pred > len(arr):
    #             break
    #         X.append(arr[i:i+length,:])
    #         if return_sequences:
    #             y.append(res[i:i+length])
    #         else:
    #             y.append(res[i+length-1])
    #         i += skip
    #     X = array(X)
    #     y = array(y)
    #     if len(shape(X)) == 1:
    #         X = np.expand_dims(X,axis=len(shape(X)))
    #     if return_sequences and len(shape(y)) == 1:
    #         y = np.expand_dims(y,axis=len(shape(y)))
    #     return X,y

   
    # def array_to_path_and_next(self,arr):
    #     length = self.conf['model']['length']
    #     skip = self.conf['model']['skip']
    #     X = []
    #     y = []
    #     i = 0
    #     while True:
    #         pred = i+length
    #         if pred >= len(arr):
    #             break
    #         X.append(arr[i:i+length])
    #         y.append(arr[i+length])
    #         i += skip
    #     X = array(X)
    #     X = np.expand_dims(X,axis=len(shape(X)))
    #     return X,array(y)

    
    # def array_to_path(self,arr):
    #     length = self.conf['model']['length']
    #     skip = self.conf['model']['skip']
    #     X = []
    #     i = 0
    #     while True:
    #         pred = i+length
    #         if pred > len(arr):
    #             break
    #         X.append(arr[i:i+length,:])
    #         i += skip
    #     X = array(X)
    #     if len(shape(X)) == 1:
    #         X = np.expand_dims(X,axis=len(shape(X)))
    #     return X

    
    #unused: handling sequences of shots        
    # def load_shots_as_X_y(conf,shots,verbose=False,stateful=True,prediction_mode=False):
    #     X,y = zip(*[load_shot_as_X_y(conf,shot,verbose,stateful,prediction_mode) for shot in shots])
    #     return vstack(X),hstack(y)

    # def load_shots_as_X_y_list(conf,shots,verbose=False,stateful=True,prediction_mode=False):
    #     return [load_shot_as_X_y(conf,shot,verbose,stateful,prediction_mode) for shot in shots]
