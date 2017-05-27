import plasma.models.targets as t
import getpass
import yaml

def parameters(input_file):
    """Parse yaml file of configuration parameters."""

    with open(input_file, 'r') as yaml_file:
        params = yaml.load(yaml_file)


        params['user_name'] = getpass.getuser()
        output_path = params['fs_path'] + "/" + params['user_name']
        base_path = output_path

        params['paths']['base_path'] = base_path
        params['paths']['signal_prepath'] = base_path + params['paths']['signal_prepath']
        params['paths']['shot_list_dir'] = base_path + params['paths']['shot_list_dir']
        params['paths']['output_path'] = output_path
        params['paths']['processed_prepath'] = output_path +'/processed_shots/'
        params['paths']['results_prepath'] = output_path + '/results/'
	if params['training']['hyperparam_tuning']:
            params['paths']['saved_shotlist_path'] = './normalization/shot_lists.npz'
            params['paths']['normalizer_path'] = './normalization/normalization.npz'
            params['paths']['model_save_path'] = './model_checkpoints/'
            params['paths']['csvlog_save_path'] = './csv_logs/'
	else:
            params['paths']['saved_shotlist_path'] = output_path +'/normalization/shot_lists.npz'
            params['paths']['normalizer_path'] = output_path + '/normalization/normalization.npz'
            params['paths']['model_save_path'] = output_path + '/model_checkpoints/'
            params['paths']['csvlog_save_path'] = output_path + '/csv_logs/'
        params['paths']['tensorboard_save_path'] = output_path + params['paths']['tensorboard_save_path']

        if params['target'] == 'hinge':
            params['data']['target'] = t.HingeTarget
        elif params['target'] == 'maxhinge':
            t.MaxHingeTarget.fac = params['data']['positive_example_penalty']
            params['data']['target'] = t.MaxHingeTarget
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
