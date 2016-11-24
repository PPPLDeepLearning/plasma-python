import plasma.models.targets as t
from plasma.jet_signals import signals_dirs, signals_masks 
import getpass
#currently need to change above import when switching machines

#paths#
fs_path = '/tigress'
user_name = 'jk7/data'#getpass.getuser()
output_path = fs_path + "/" + user_name
base_path = output_path #decentralized signal data on Tiger
target = t.HingeTarget

conf = {
    'paths': {
        'base_path' : base_path,
        'signal_prepath' : base_path + '/signal_data/jet/',
        'signals_dirs' : [[signal] for i,group in enumerate(signals_dirs) for j,signal in enumerate(group) if signals_masks[i][j]],
#        'signals_dirs' : signals_dirs,
        'signals_masks' : signals_masks,
        'shot_files' : ['CWall_clear.txt','CFC_unint.txt'],#['mixed_list1.txt'],#['short_list.txt'],#['CWall_clear.txt','CFC_unint.txt'],#['mixed_list1.txt',long_list_C.txt','short_list.txt','BeWall_clear.txt']
        'shot_files_test' : ['BeWall_clear.txt','ILW_unint.txt'] ,#[],#['BeWall_clear.txt','ILW_unint.txt'],
        'shot_list_dir' : base_path + '/shot_lists/',
        'output_path' : output_path,
        #Each of these output paths should be created before running code
        #In the future, have better error checking for directories
        'processed_prepath' : output_path +'/processed_shots/',
        'normalizer_path' : output_path + '/normalization/normalization.npz',
        'results_prepath' : output_path + '/results/',
        'model_save_path' : output_path + '/model_checkpoints/'
   },

   'data': {
        'T_min_warn' : 30, #number of miliseconds (multiples of dt) before disruption that aren't used for training.
        'recompute' : False,
        'recompute_normalization' : False,
        #'recompute_minmax' : False
        'num_signals' : sum([group.count(True) for i,group in enumerate(signals_masks)]),
        'current_index' : 0, #specifies which of the signals in the signals_dirs order contains the plasma current info
        'plotting' : False,
        #train/validate split
        #how many shots to use
        'use_shots' : 200000, #100,#200000,
        #normalization timescale
        'dt' : 0.001,
        #maximum TTD considered
        'T_max' : 1000.0,
        'T_warning' : 1.0, #The shortest works best so far: less overfitting. log TTd prediction also works well. 0.5 better than 0.2
        'current_thresh' : 750000,
        'current_end_thresh' : 10000,
        'window_decay' : 2, #the characteristic decay length of the decaying moving average window
        'window_size' : 10, #the width of the actual window
        'target' : target,
        'normalizer' : 'var',           #TODO optimize
   },

   'model': {
        #length of LSTM memory
        'pred_length' : 200,
        'pred_batch_size' : 128,
        'length' : 128,                     #TODO optimize
        'skip' : 1,
        #hidden layer size
        'rnn_size' : 300,                   #TODO optimize
        #size 100 slight overfitting, size 20 no overfitting. 200 is not better than 100. Prediction much better with size 100, size 20 cannot capture the data.
        'rnn_type' : 'LSTM',
        'rnn_layers' : 3,                   #TODO optimize
        # 'output_activation' : target.activation,
        'optimizer' : 'adam', #have not found a difference yet
        'clipnorm' : 10.0,
        'regularization' : 0.0,#5e-6,#0.00001,
        # 'loss' : target.loss, #binary crossentropy performs slightly better?
        'lr' : 5e-5,#1e-4 is too high, 5e-7 is too low. 5e-5 seems best at 256 batch size, full dataset and ~10 epochs, and lr decay of 0.90. 1e-4 also works well if we decay a lot (i.e ~0.7 or more)
        'lr_decay' : 0.9,
        'stateful' : True,
        'return_sequences' : True,
        'dropout_prob' : 0.3,
	'warmup_steps' : 0, #only relevant if we want to do mpi training. The number of steps with a single replica
    },

    'training': {
        'as_array_of_shots':True,
        'shuffle_training' : True,
        'train_frac' : 0.5,
        'validation_frac' : 0.05,
        'batch_size' : 256, #100
        'max_patch_length' : 100000, #THIS WAS THE CULPRIT FOR NO TRAINING! Lower than 1000 performs very poorly
        'num_shots_at_once' : 200, #How many shots are we loading at once?
        'num_epochs' : 3,
        'use_mock_data' : False,
        'data_parallel' : False,
   },
}
