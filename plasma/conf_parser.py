import plasma.models.targets as t
import getpass
import yaml

def parameters(input_file):
    """Parse yaml file of configuration parameters."""

    with open(input_file, 'r') as yaml_file:
        params = yaml.load(yaml_file)

        signals_dirs = params['paths']['signals_dirs']
	


        #signal masks
        to_mask = params['paths']['signals_masks']
        to_mask = [item for sublist in to_mask for item in sublist]
        signals_masks = [[True if sig not in to_mask else False for sig in group] for group in signals_dirs]

        #positivity masks
        to_positivity_mask = params['paths']['positivity_mask']
        to_positivity_mask = [item for sublist in to_positivity_mask for item in sublist]
        positivity_mask = [[True if sig not in to_positivity_mask else False for sig in group] for group in signals_dirs]

        #plot masks
        to_plot_mask = params['plots']['plot_masks']
        to_plot_mask = [item for sublist in to_plot_mask for item in sublist]
        plot_mask = [[True if sig not in to_plot_mask else False for sig in group] for group in signals_dirs]

        params['user_name'] = getpass.getuser()
        output_path = params['fs_path'] + "/" + params['user_name']
        base_path = output_path

        params['paths']['base_path'] = base_path
        params['paths']['signal_prepath'] = base_path + params['paths']['signal_prepath']
        params['paths']['signals_masks'] = signals_masks
        params['paths']['positivity_mask'] = positivity_mask
        params['paths']['shot_list_dir'] = base_path + params['paths']['shot_list_dir']
        params['paths']['output_path'] = output_path
        params['paths']['processed_prepath'] = output_path +'/processed_shots/'
        params['paths']['normalizer_path'] = output_path + '/normalization/normalization.npz'
        params['paths']['results_prepath'] = output_path + '/results/'
        params['paths']['model_save_path'] = output_path + '/model_checkpoints/'
        params['paths']['callback_save_path'] = output_path + '/callback_logs/'

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

        params['plots']['plot_masks'] = plot_mask
 
        #params['model']['output_activation'] = params['data']['target'].activation
        #binary crossentropy performs slightly better?
        #params['model']['loss'] = params['data']['target'].loss

    return params
