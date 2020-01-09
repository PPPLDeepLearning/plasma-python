from __future__ import print_function
import plasma.global_vars as g
from plasma.primitives.shots import ShotListFiles
import data.signals as sig
from plasma.utils.hashing import myhash_signals
# from data.signals import (
#     all_signals, fully_defined_signals_1D,
#     jet, d3d)  # nstx
import getpass
import yaml


def parameters(input_file):
    """Parse yaml file of configuration parameters."""
    # TODO(KGF): the following line imports TensorFlow as a Keras backend
    # by default (absent env variable KERAS_BACKEND and/or config file
    # $HOME/.keras/keras.json) "from plasma.conf import conf"
    # via "import keras.backend as K" in targets.py
    from plasma.models.targets import (
        HingeTarget, MaxHingeTarget, BinaryTarget,
        TTDTarget, TTDInvTarget, TTDLinearTarget
        )
    with open(input_file, 'r') as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.SafeLoader)
        params['user_name'] = getpass.getuser()
        output_path = params['fs_path'] + "/" + params['user_name']
        base_path = output_path

        params['paths']['base_path'] = base_path
        if isinstance(params['paths']['signal_prepath'], list):
            g.print_unique('Reading from multiple data folders!')
            params['paths']['signal_prepath'] = [
                base_path + s for s in params['paths']['signal_prepath']]
        else:
            params['paths']['signal_prepath'] = (
                base_path + params['paths']['signal_prepath'])
        params['paths']['shot_list_dir'] = (
            base_path + params['paths']['shot_list_dir'])
        params['paths']['output_path'] = output_path
        # See notes in data/signals.py for details on signal tols relative to
        # t_disrupt. The following 2x dataset definitions permit progressively
        # worse signal quality when preprocessing the shots and omitting some
        if params['paths']['data'] == 'd3d_data_max_tol':
            # let signals terminate up to 29 ms before t_disrupt on D3D
            h = myhash_signals(sig.all_signals_max_tol.values())
        elif params['paths']['data'] == 'd3d_data_garbage':
            # let up to 3x signals disappear at any time before t_disrupt
            # (and NaNs?)
            # -----
            # temp workaround for identical signal dictionary (but different
            # omit criteria in shots.py Shot.get_signals_and_times_from_file())
            # ---> 2x hash int
            # TODO(KGF): not robust; create reproducible specification and
            # recording of signal filtering procedure
            h = myhash_signals(sig.all_signals_max_tol.values())*2
        else:
            h = myhash_signals(sig.all_signals.values())
        params['paths']['global_normalizer_path'] = (
            output_path
            + '/normalization/normalization_signal_group_{}.npz'.format(h))
        if params['training']['hyperparam_tuning']:
            # params['paths']['saved_shotlist_path'] =
            # './normalization/shot_lists.npz'
            params['paths']['normalizer_path'] = (
                './normalization/normalization_signal_group_{}.npz'.format(h))
            params['paths']['model_save_path'] = './model_checkpoints/'
            params['paths']['csvlog_save_path'] = './csv_logs/'
            params['paths']['results_prepath'] = './results/'
        else:
            # params['paths']['saved_shotlist_path'] = output_path +
            # '/normalization/shot_lists.npz'
            params['paths']['normalizer_path'] = (
                params['paths']['global_normalizer_path'])
            params['paths']['model_save_path'] = (output_path
                                                  + '/model_checkpoints/')
            params['paths']['csvlog_save_path'] = output_path + '/csv_logs/'
            params['paths']['results_prepath'] = output_path + '/results/'
        params['paths']['tensorboard_save_path'] = (
            output_path + params['paths']['tensorboard_save_path'])
        params['paths']['saved_shotlist_path'] = (
            params['paths']['base_path'] + '/processed_shotlists/'
            + params['paths']['data']
            + '/shot_lists_signal_group_{}.npz'.format(h))
        params['paths']['processed_prepath'] = (
            output_path + '/processed_shots/' + 'signal_group_{}/'.format(h))
        # ensure shallow model has +1 -1 target.
        if params['model']['shallow'] or params['target'] == 'hinge':
            params['data']['target'] = HingeTarget
        elif params['target'] == 'maxhinge':
            MaxHingeTarget.fac = params['data']['positive_example_penalty']
            params['data']['target'] = MaxHingeTarget
        elif params['target'] == 'binary':
            params['data']['target'] = BinaryTarget
        elif params['target'] == 'ttd':
            params['data']['target'] = TTDTarget
        elif params['target'] == 'ttdinv':
            params['data']['target'] = TTDInvTarget
        elif params['target'] == 'ttdlinear':
            params['data']['target'] = TTDLinearTarget
        else:
            # TODO(KGF): "Target" base class is unused here
            g.print_unique('Unknown type of target. Exiting')
            exit(1)

        # params['model']['output_activation'] =
        # params['data']['target'].activation
        # binary crossentropy performs slightly better?
        # params['model']['loss'] = params['data']['target'].loss

        # signals
        if params['paths']['data'] in ['d3d_data_max_tol', 'd3d_data_garbage']:
            params['paths']['all_signals_dict'] = sig.all_signals_max_tol
        else:
            params['paths']['all_signals_dict'] = sig.all_signals

        # assert order
        # q95, li, ip, lm, betan, energy, dens, pradcore, pradedge, pin,
        # pechin, torquein, ipdirect, etemp_profile, edens_profile

        # shot lists
        jet_carbon_wall = ShotListFiles(
            sig.jet, params['paths']['shot_list_dir'],
            ['CWall_clear.txt', 'CFC_unint.txt'], 'jet carbon wall data')
        jet_iterlike_wall = ShotListFiles(
            sig.jet, params['paths']['shot_list_dir'],
            ['ILW_unint.txt', 'BeWall_clear.txt'], 'jet iter like wall data')
        jet_iterlike_wall_late = ShotListFiles(
            sig.jet, params['paths']['shot_list_dir'],
            ['ILW_unint_late.txt', 'ILW_clear_late.txt'],
            'Late jet iter like wall data')
        # jet_iterlike_wall_full = ShotListFiles(
        #     sig.jet, params['paths']['shot_list_dir'],
        #     ['ILW_unint_full.txt', 'ILW_clear_full.txt'],
        #     'Full jet iter like wall data')

        jenkins_jet_carbon_wall = ShotListFiles(
            sig.jet, params['paths']['shot_list_dir'],
            ['jenkins_CWall_clear.txt', 'jenkins_CFC_unint.txt'],
            'Subset of jet carbon wall data for Jenkins tests')
        jenkins_jet_iterlike_wall = ShotListFiles(
            sig.jet, params['paths']['shot_list_dir'],
            ['jenkins_ILW_unint.txt', 'jenkins_BeWall_clear.txt'],
            'Subset of jet iter like wall data for Jenkins tests')

        jet_full = ShotListFiles(
            sig.jet, params['paths']['shot_list_dir'],
            ['ILW_unint.txt', 'BeWall_clear.txt', 'CWall_clear.txt',
             'CFC_unint.txt'], 'jet full data')

        # d3d_10000 = ShotListFiles(
        #     sig.d3d, params['paths']['shot_list_dir'],
        #     ['d3d_clear_10000.txt', 'd3d_disrupt_10000.txt'],
        #     'd3d data 10000 ND and D shots')
        # d3d_1000 = ShotListFiles(
        #     sig.d3d, params['paths']['shot_list_dir'],
        #     ['d3d_clear_1000.txt', 'd3d_disrupt_1000.txt'],
        #     'd3d data 1000 ND and D shots')
        # d3d_100 = ShotListFiles(
        #     sig.d3d, params['paths']['shot_list_dir'],
        #     ['d3d_clear_100.txt', 'd3d_disrupt_100.txt'],
        #     'd3d data 100 ND and D shots')
        d3d_full = ShotListFiles(
            sig.d3d, params['paths']['shot_list_dir'],
            ['d3d_clear_data_avail.txt', 'd3d_disrupt_data_avail.txt'],
            'd3d data since shot 125500')  # to 168555
        # superset of d3d_full added in 2019 from C. Rea:
        d3d_full_2019 = ShotListFiles(
            sig.d3d, params['paths']['shot_list_dir'],
            ['d3d_clear_since_2016.txt', 'd3d_disrupt_since_2016.txt'],
            'd3d data since shot 125500')  # to 180847
        d3d_jenkins = ShotListFiles(
            sig.d3d, params['paths']['shot_list_dir'],
            ['jenkins_d3d_clear.txt', 'jenkins_d3d_disrupt.txt'],
            'Subset of d3d data for Jenkins test')

        # TODO(KGF): currently unused shot list files in project directory
        # /tigress/FRNN/shot_lists/:
        # d3d_clear.txt : 40560, 168554
        # d3d_disrupt   : 100000, 168555

        # TODO(KGF): should /tigress/FRNN/shot_lists/ be organized into subdirs
        # like the original repo directory data/shot_lists/d3d/, jet/, nstx/ ?

        # d3d_jb_full = ShotListFiles(
        #     sig.d3d, params['paths']['shot_list_dir'],
        #     ['shotlist_JaysonBarr_clear.txt',
        #      'shotlist_JaysonBarr_disrupt.txt'],
        #     'd3d shots since 160000-170000')

        # nstx_full = ShotListFiles(
        #     nstx, params['paths']['shot_list_dir'],
        #     ['disrupt_nstx.txt'], 'nstx shots (all are disruptive')
        # ==================
        # JET DATASETS
        # ==================
        if params['paths']['data'] == 'jet_all':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = sig.jet_signals
        elif params['paths']['data'] == 'jet_0D':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = sig.jet_signals_0D
        elif params['paths']['data'] == 'jet_1D':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = sig.jet_signals_1D
        elif params['paths']['data'] == 'jet_late':
            params['paths']['shot_files'] = [jet_iterlike_wall_late]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = sig.jet_signals
        elif params['paths']['data'] == 'jet_carbon_to_late_0D':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [jet_iterlike_wall_late]
            params['paths']['use_signals_dict'] = sig.jet_signals_0D
        elif params['paths']['data'] == 'jet_temp_profile':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = {
                'etemp_profile': sig.etemp_profile}
        elif params['paths']['data'] == 'jet_dens_profile':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = {
                'edens_profile': sig.edens_profile}
        elif params['paths']['data'] == 'jet_carbon_all':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = sig.jet_signals
        elif params['paths']['data'] == 'jet_mixed_all':
            params['paths']['shot_files'] = [jet_full]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = sig.jet_signals
        elif params['paths']['data'] == 'jenkins_jet':
            params['paths']['shot_files'] = [jenkins_jet_carbon_wall]
            params['paths']['shot_files_test'] = [jenkins_jet_iterlike_wall]
            params['paths']['use_signals_dict'] = sig.jet_signals
        # JET data but with fully defined signals
        elif params['paths']['data'] == 'jet_fully_defined':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = sig.fully_defined_signals
        # JET data but with fully defined signals
        elif params['paths']['data'] == 'jet_fully_defined_0D':
            params['paths']['shot_files'] = [jet_carbon_wall]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = sig.fully_defined_signals_0D
        # ==================
        # D3D DATASETS
        # ==================
        elif params['paths']['data'] == 'd3d_all':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = {
                'q95': sig.q95,
                'li': sig.li,
                'ip': sig.ip,
                'lm': sig.lm,
                'betan': sig.betan,
                'energy': sig.energy,
                'dens': sig.dens,
                'pradcore': sig.pradcore,
                'pradedge': sig.pradedge,
                'pin': sig.pin,
                'torquein': sig.torquein,
                'ipdirect': sig.ipdirect,
                'iptarget': sig.iptarget,
                'iperr': sig.iperr,
                'etemp_profile': sig.etemp_profile,
                'edens_profile': sig.edens_profile,
            }
        elif params['paths']['data'] in ['d3d_data_max_tol',
                                         'd3d_data_garbage']:
            params['paths']['shot_files'] = [d3d_full_2019]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = {
                'q95t': sig.q95t,
                'lit': sig.lit,
                'ipt': sig.ipt,
                'lmt': sig.lmt,
                'betant': sig.betant,
                'energyt': sig.energyt,
                'denst': sig.denst,
                'pradcoret': sig.pradcoret,
                'pradedget': sig.pradedget,
                'pint': sig.pint,
                'torqueint': sig.torqueint,
                'ipdirectt': sig.ipdirectt,
                'iptargett': sig.iptargett,
                'iperrt': sig.iperrt,
                'etemp_profilet': sig.etemp_profilet,
                'edens_profilet': sig.edens_profilet,
            }
        elif params['paths']['data'] == 'd3d_2019':
            params['paths']['shot_files'] = [d3d_full_2019]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = {
                'q95': sig.q95,
                'li': sig.li,
                'ip': sig.ip,
                'lm': sig.lm,
                'betan': sig.betan,
                'energy': sig.energy,
                'dens': sig.dens,
                'pradcore': sig.pradcore,
                'pradedge': sig.pradedge,
                'pin': sig.pin,
                'torquein': sig.torquein,
                'ipdirect': sig.ipdirect,
                'iptarget': sig.iptarget,
                'iperr': sig.iperr,
                'etemp_profile': sig.etemp_profile,
                'edens_profile': sig.edens_profile,
            }
        elif params['paths']['data'] == 'd3d_1D':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = {
                'ipdirect': sig.ipdirect,
                'etemp_profile': sig.etemp_profile,
                'edens_profile': sig.edens_profile,
            }
        elif params['paths']['data'] == 'd3d_all_profiles':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = {
                'ipdirect': sig.ipdirect,
                'etemp_profile': sig.etemp_profile,
                'edens_profile': sig.edens_profile,
                'itemp_profile': sig.itemp_profile,
                'zdens_profile': sig.zdens_profile,
                'trot_profile': sig.trot_profile,
                'pthm_profile': sig.pthm_profile,
                'neut_profile': sig.neut_profile,
                'q_profile': sig.q_profile,
                'bootstrap_current_profile': sig.bootstrap_current_profile,
                'q_psi_profile': sig.q_psi_profile,
            }
        elif params['paths']['data'] == 'd3d_0D':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = {
                'q95': sig.q95,
                'li': sig.li,
                'ip': sig.ip,
                'lm': sig.lm,
                'betan': sig.betan,
                'energy': sig.energy,
                'dens': sig.dens,
                'pradcore': sig.pradcore,
                'pradedge': sig.pradedge,
                'pin': sig.pin,
                'torquein': sig.torquein,
                'ipdirect': sig.ipdirect,
                'iptarget': sig.iptarget,
                'iperr': sig.iperr,
            }
        # TODO(KGF): rename. Unlike JET, there are probably differences between
        # sig.d3d_signals and the manually-defined sigs in above d3d_all
        # elif params['paths']['data'] == 'd3d_all':
        #     params['paths']['shot_files'] = [d3d_full]
        #     params['paths']['shot_files_test'] = []
        #     params['paths']['use_signals_dict'] = sig.d3d_signals
        elif params['paths']['data'] == 'jenkins_d3d':
            params['paths']['shot_files'] = [d3d_jenkins]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = {
                'q95': sig.q95,
                'li': sig.li,
                'ip': sig.ip,
                'lm': sig.lm,
                'betan': sig.betan,
                'energy': sig.energy,
                'dens': sig.dens,
                'pradcore': sig.pradcore,
                'pradedge': sig.pradedge,
                'pin': sig.pin,
                'torquein': sig.torquein,
                'ipdirect': sig.ipdirect,
                'iptarget': sig.iptarget,
                'iperr': sig.iperr,
                'etemp_profile': sig.etemp_profile,
                'edens_profile': sig.edens_profile,
            }
        # jet data but with fully defined signals
        elif params['paths']['data'] == 'd3d_fully_defined':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = sig.fully_defined_signals
        # jet data but with fully defined signals
        elif params['paths']['data'] == 'd3d_fully_defined_0D':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = sig.fully_defined_signals_0D
        elif params['paths']['data'] == 'd3d_temp_profile':
            # jet data but with fully defined signals
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = {
                'etemp_profile': sig.etemp_profile}  # fully_defined_signals_0D
        elif params['paths']['data'] == 'd3d_dens_profile':
            # jet data but with fully defined signals
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = []
            params['paths']['use_signals_dict'] = {
                'edens_profile': sig.edens_profile}  # fully_defined_signals_0D
        # ======================
        # CROSS-MACHINE DATASETS
        # ======================
        elif params['paths']['data'] == 'jet_to_d3d_all':
            params['paths']['shot_files'] = [jet_full]
            params['paths']['shot_files_test'] = [d3d_full]
            params['paths']['use_signals_dict'] = sig.fully_defined_signals
        elif params['paths']['data'] == 'd3d_to_jet_all':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = sig.fully_defined_signals
        elif params['paths']['data'] == 'd3d_to_late_jet':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = [jet_iterlike_wall_late]
            params['paths']['use_signals_dict'] = sig.fully_defined_signals
        elif params['paths']['data'] == 'jet_to_d3d_0D':
            params['paths']['shot_files'] = [jet_full]
            params['paths']['shot_files_test'] = [d3d_full]
            params['paths']['use_signals_dict'] = sig.fully_defined_signals_0D
        elif params['paths']['data'] == 'd3d_to_jet_0D':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = sig.fully_defined_signals_0D
        elif params['paths']['data'] == 'jet_to_d3d_1D':
            params['paths']['shot_files'] = [jet_full]
            params['paths']['shot_files_test'] = [d3d_full]
            params['paths']['use_signals_dict'] = sig.fully_defined_signals_1D
        elif params['paths']['data'] == 'd3d_to_jet_1D':
            params['paths']['shot_files'] = [d3d_full]
            params['paths']['shot_files_test'] = [jet_iterlike_wall]
            params['paths']['use_signals_dict'] = sig.fully_defined_signals_1D

        else:
            g.print_unique("Unknown dataset {}".format(
                params['paths']['data']))
            exit(1)

        if len(params['paths']['specific_signals']):
            for s in params['paths']['specific_signals']:
                if s not in params['paths']['use_signals_dict'].keys():
                    g.print_unique(
                        "Signal {} is not fully defined for {} machine. ",
                        "Skipping...".format(
                            s, params['paths']['data'].split("_")[0]))
            params['paths']['specific_signals'] = list(
                filter(
                    lambda x: x in params['paths']['use_signals_dict'].keys(),
                    params['paths']['specific_signals']))
            selected_signals = {k: params['paths']['use_signals_dict'][k]
                                for k in params['paths']['specific_signals']}
            params['paths']['use_signals'] = sort_by_channels(
                list(selected_signals.values()))
        else:
            # default case
            params['paths']['use_signals'] = sort_by_channels(
                list(params['paths']['use_signals_dict'].values()))

        params['paths']['all_signals'] = sort_by_channels(
            list(params['paths']['all_signals_dict'].values()))

        g.print_unique("Selected signals (determines which signals are used"
                       + " for training):\n{}".format(
                           params['paths']['use_signals']))
        params['paths']['shot_files_all'] = (
            params['paths']['shot_files'] + params['paths']['shot_files_test'])
        params['paths']['all_machines'] = list(
            set([file.machine for file in params['paths']['shot_files_all']]))

        # type assertations
        assert (isinstance(params['data']['signal_to_augment'], str)
                or isinstance(params['data']['signal_to_augment'], None))
        assert isinstance(params['data']['augment_during_training'], bool)

    return params


def sort_by_channels(list_of_signals):
    # make sure 1D signals come last! This is necessary for model builder.
    return sorted(list_of_signals, key=lambda x: x.num_channels)
