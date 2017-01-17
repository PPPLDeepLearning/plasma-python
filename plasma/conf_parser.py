import plasma.models.targets as t
import getpass
import yaml

def parameters(input_file):
    """Parse yaml file of configuration parameters."""

    signals_dirs = yaml.load(open(input_file, 'r'))['paths']['signals_dirs']

    #signal masks
    signals_masks = [[True]*len(sig_list) for sig_list in signals_dirs]
    signals_masks[4] = [True]
    signals_masks[7] = [False]*len(signals_dirs[7])
    signals_masks[7][1] = True

    #positivity masks
    positivity_mask = [[True]*len(sig_list) for sig_list in signals_dirs]
    positivity_mask[0] = [False]
    positivity_mask[4] = [False]

    with open(input_file, 'r') as yaml_file:
        params = yaml.load(yaml_file)

        params['user_name'] = getpass.getuser()
        output_path = params['fs_path'] + "/" + params['user_name']
        base_path = output_path

        params['paths']['base_path'] = base_path
        params['paths']['signal_prepath'] = base_path + '/signal_data/jet/'
        params['paths']['signals_masks'] = signals_masks
        params['paths']['positivity_mask'] = positivity_mask
        params['paths']['shot_list_dir'] = base_path + '/shot_lists/'
        params['paths']['output_path'] = output_path
        params['paths']['processed_prepath'] = output_path +'/processed_shots/'
        params['paths']['normalizer_path'] = output_path + '/normalization/normalization.npz'
        params['paths']['results_prepath'] = output_path + '/results/'
        params['paths']['model_save_path'] = output_path + '/model_checkpoints/'

        params['data']['num_signals'] = sum([sum([1 for predicate in subl if predicate]) for subl in signals_masks])
        if params['target'] == 'hinge':
            params['data']['target'] = t.HingeTarget
        elif params['target'] == 'binary':
            params['data']['target'] = t.BinaryTarget
        elif params['target'] == 'ttd':
            params['data']['target'] = t.TTDTarget
        elif params['target'] == 'ttdlinear':
            params['data']['target'] = t.TTDLinearTarget
        else:
            print('Unkown type of target. Exiting')
            exit(1)
 
        #params['model']['output_activation'] = params['data']['target'].activation
        #binary crossentropy performs slightly better?
        #params['model']['loss'] = params['data']['target'].loss

    return params
