import plasma.models.targets as t
from plasma.primitives.shots import ShotListFiles
from data.signals import *

import getpass
import uuid
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
        if params['training']['hyperparam_tuning']:
            params['paths']['saved_shotlist_path'] = './normalization/shot_lists.npz'
            params['paths']['normalizer_path'] = './normalization/normalization.npz'
            params['paths']['model_save_path'] = './model_checkpoints/'
            params['paths']['csvlog_save_path'] = './csv_logs/'
            params['paths']['results_prepath'] = './results/'
        else:
            params['paths']['saved_shotlist_path'] = output_path +'/normalization/shot_lists.npz'
            params['paths']['normalizer_path'] = output_path + '/normalization/normalization.npz'
            params['paths']['model_save_path'] = output_path + '/model_checkpoints/'
            params['paths']['csvlog_save_path'] = output_path + '/csv_logs/'
            params['paths']['results_prepath'] = output_path + '/results/'
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

        #signals
        params['paths']['all_signals_dict'] = all_signals
        #assert order q95,li,ip,lm,betan,energy,dens,pradcore,pradedge,pin,pechin,torquein,ipdirect,etemp_profile,edens_profile

        #shot lists
        jet_carbon_wall = ShotListFiles(jet,params['paths']['shot_list_dir'],['CWall_clear.txt','CFC_unint.txt'],'jet carbon wall data')
        jet_iterlike_wall = ShotListFiles(jet,params['paths']['shot_list_dir'],['ILW_unint.txt','BeWall_clear.txt'],'jet iter like wall data')
        jet_full = ShotListFiles(jet,params['paths']['shot_list_dir'],['ILW_unint.txt','BeWall_clear.txt','CWall_clear.txt','CFC_unint.txt'],'jet full data')

        d3d_10000 = ShotListFiles(d3d,params['paths']['shot_list_dir'],['d3d_clear_10000.txt','d3d_disrupt_10000.txt'],'d3d data 10000 ND and D shots')
        d3d_1000 = ShotListFiles(d3d,params['paths']['shot_list_dir'],['d3d_clear_1000.txt','d3d_disrupt_1000.txt'],'d3d data 1000 ND and D shots')
        d3d_100 = ShotListFiles(d3d,params['paths']['shot_list_dir'],['d3d_clear_100.txt','d3d_disrupt_100.txt'],'d3d data 100 ND and D shots')
        d3d_full = ShotListFiles(d3d,params['paths']['shot_list_dir'],['d3d_clear_data_avail.txt','d3d_disrupt_data_avail.txt'],'d3d data since shot 125500')
        d3d_jb_full = ShotListFiles(d3d,params['paths']['shot_list_dir'],['shotlist_JaysonBarr_clear.txt','shotlist_JaysonBarr_disrupt.txt'],'d3d shots since 160000-170000')

        nstx_full = ShotListFiles(nstx,params['paths']['shot_list_dir'],['disrupt_nstx.txt'],'nstx shots (all are disruptive')

        apply_positivity_test = ShotListFiles(jet,params['paths']['shot_list_dir'],['apply_positivity_test.txt'],'a couple of negative density test shots')
 
        if params['paths']['data'] == 'jet_data':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = jet_signals
        elif params['paths']['data'] == 'jet_carbon_data':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = jet_signals
        elif params['paths']['data'] == 'd3d_data':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = [] 
            #make sure all 1D signals appear last!
            params['paths']['use_signals_dict'] = {'q95':q95,'li':li,'ip':ip,'lm':lm,'betan':betan,'energy':energy,'dens':dens,'pradcore':pradcore,
'pradedge':pradedge,'pin':pin,'pechin':pechin,'torquein':torquein,'ipdirect':ipdirect} #'etemp_profile':etemp_profile,'edens_profile'}
            #[q95,li,ip,lm,betan,energy,dens,pradcore,pradedge,pin,pechin,torquein,ipdirect,etemp_profile,edens_profile][:-2]

        elif params['paths']['data'] == 'apply_positivity_test':
            params['paths']['shot_files'] = [apply_positivity_test]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = jet_signals

        elif params['paths']['data'] == 'jet_to_d3d_data':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [d3d_full]
            params['paths']['use_signals_dict'] = fully_defined_signals
        elif params['paths']['data'] == 'd3d_to_jet_data':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = fully_defined_signals
        else: 
            print("Unkown data set {}".format(params['paths']['data']))
            exit(1)

        if len(params['paths']['specific_signals']):
            for sig in params['paths']['specific_signals']:
                if sig not in params['paths']['use_signals_dict'].keys():
                    print("Signal {} is not fully defined for {} machine. Skipping...".format(sig,params['paths']['data'].split("_")[0]))
            params['paths']['specific_signals'] = list(filter(lambda x: x in params['paths']['use_signals_dict'].keys(), params['paths']['specific_signals']))
            selected_signals = {k: params['paths']['use_signals_dict'][k] for k in params['paths']['specific_signals']}
            params['paths']['use_signals'] = list(selected_signals.values())

            selected_signals = {k: params['paths']['all_signals_dict'][k] for k in params['paths']['specific_signals']}
            params['paths']['all_signals'] = list(selected_signals.values())
        else:
            #default case
            params['paths']['use_signals'] = list(params['paths']['use_signals_dict'].values())
            params['paths']['all_signals'] = list(params['paths']['all_signals_dict'].values())

        print("Selected signals {}".format(params['paths']['use_signals']))

        params['paths']['shot_files_all'] = params['paths']['shot_files']+params['paths']['shot_files_test']
        params['paths']['all_machines'] = list(set([file.machine for file in params['paths']['shot_files_all']]))

        #type assertations
        assert type(params['data']['signal_to_augment']) == str or type(params['data']['signal_to_augment']) == None
        assert type(params['data']['augment_during_training']) == bool

    return params
