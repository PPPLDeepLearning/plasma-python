import plasma.models.targets as t
import getpass
import yaml

def parameters(input_file):
    """Parse yaml file of configuration parameters."""


    #signal dirs
    signals_dirs = [['jpf/da/c2-ipla'], # Plasma Current [A]
                ['jpf/da/c2-loca'], # Mode Lock Amplitude [A]
                ['jpf/db/b5r-ptot>out'], #Radiated Power [W]
                ['jpf/gs/bl-li<s'], #Plasma Internal Inductance
                ['jpf/gs/bl-fdwdt<s'], #Stored Diamagnetic Energy (time derivative) [W] Might cause a lot of false positives!
                ['jpf/gs/bl-ptot<s'], #total input power [W]
                ['jpf/gs/bl-wmhd<s']] #total stored diamagnetic energy

    #density signals [m^-2]
    #4 vertical channels and 4 horizontal channels
    signals_dirs += [['jpf/df/g1r-lid:{:03d}'.format(i) for i in range(2,9)]]

    #signal masks
    signals_masks = [[True]*len(sig_list) for sig_list in signals_dirs]
    signals_masks[4] = [True]
    signals_masks[7] = [False]*len(signals_dirs[7])
    signals_masks[7][1] = True

    #positivity masks
    positivity_mask = [[True]*len(sig_list) for sig_list in signals_dirs]
    positivity_mask[0] = [False]
    positivity_mask[4] = [False]

    #radiation signals
    #vertical signals, don't use signal 16 and 23
    # signals_dirs += ['jpf/db/b5vr-pbol:{:03d}'.format(i) for i in range(1,28) if (i != 16 and i != 23)]
    # signals_dirs += ['jpf/db/b5hr-pbol:{:03d}'.format(i) for i in range(1,24)]

    #ece temperature profiles
    #temperature of channel i vs time
    # signals_dirs += ['ppf/kk3/te{:02d}'.format(i) for i in range(1,97)]
    #radial position of channel i mapped onto midplane vs time
    # signals_dirs += ['ppf/kk3/rc{:02d}'.format(i) for i in range(1,97)]

    ####radial position of channel i vs time
    ####signal_paths += ['ppf/kk3/ra{:02d}'.format(i) for i in range(1,97)]

    with open(input_file, 'r') as yaml_file:
        params = yaml.load(yaml_file)

        params['user_name'] = getpass.getuser()
        output_path = params['fs_path'] + "/" + params['user_name']
        base_path = output_path

        params['paths']['base_path'] = base_path
        params['paths']['signal_prepath'] = base_path + '/signal_data/jet/'
        params['paths']['signals_dirs'] = signals_dirs
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
