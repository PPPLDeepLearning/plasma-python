from __future__ import print_function
import plasma.global_vars as g
import numpy as np
import sys

from plasma.primitives.data import (
    Signal, ProfileSignal, ChannelSignal, Machine
    )


def create_missing_value_filler():
    time = np.linspace(0, 100, 100)
    vals = np.zeros_like(time)
    return time, vals


def get_tree_and_tag(path):
    spl = path.split('/')
    tree = spl[0]
    tag = '\\' + spl[1]
    return tree, tag


def get_tree_and_tag_no_backslash(path):
    spl = path.split('/')
    tree = spl[0]
    tag = spl[1]
    return tree, tag


def fetch_d3d_data(signal_path, shot, c=None):
    tree, signal = get_tree_and_tag_no_backslash(signal_path)
    if tree is None:
        signal = c.get('findsig("'+signal+'",_fstree)').value
        tree = c.get('_fstree').value
    # if c is None:
        # c = MDSplus.Connection('atlas.gat.com')

    # Retrieve data
    found = False
    xdata = np.array([0])
    ydata = None
    data = np.array([0])

    # Retrieve data from MDSplus (thin)
    # first try, retrieve directly from tree andsignal
    def get_units(str):
        units = c.get('units_of('+str+')').data()
        if units == '' or units == ' ':
            units = c.get('units('+str+')').data()
        return units

    try:
        c.openTree(tree, shot)
        data = c.get('_s = '+signal).data()
        # data_units = c.get('units_of(_s)').data()
        rank = np.ndim(data)
        found = True

    except Exception as e:
        g.print_unique(e)
        sys.stdout.flush()
        pass

    # Retrieve data from PTDATA if node not found
    if not found:
        # g.print_unique("not in full path {}".format(signal))
        data = c.get('_s = ptdata2("'+signal+'",'+str(shot)+')').data()
        if len(data) != 1:
            rank = np.ndim(data)
            found = True
    # Retrieve data from Pseudo-pointname if not in ptdata
    if not found:
        # g.print_unique("not in PTDATA {}".format(signal))
        data = c.get('_s = pseudo("'+signal+'",'+str(shot)+')').data()
        if len(data) != 1:
            rank = np.ndim(data)
            found = True
    # this means the signal wasn't found
    if not found:
        g.print_unique("No such signal: {}".format(signal))
        pass

    # get time base
    if found:
        if rank > 1:
            xdata = c.get('dim_of(_s,1)').data()
            ydata = c.get('dim_of(_s)').data()
        else:
            xdata = c.get('dim_of(_s)').data()

    # MDSplus seems to return 2-D arrays transposed.  Change them back.
    if np.ndim(data) == 2:
        data = np.transpose(data)
    if np.ndim(ydata) == 2:
        ydata = np.transpose(ydata)
    if np.ndim(xdata) == 2:
        xdata = np.transpose(xdata)

    # print('   GADATA Retrieval Time : ', time.time() - t0)
    xdata = xdata*1e-3  # time is measued in ms
    return xdata, data, ydata, found


def fetch_jet_data(signal_path, shot_num, c):
    found = False
    time = np.array([0])
    ydata = None
    data = np.array([0])
    try:
        data = c.get('_sig=jet("{}/",{})'.format(signal_path, shot_num)).data()
        if np.ndim(data) == 2:
            data = np.transpose(data)
            time = c.get('_sig=dim_of(jet("{}/",{}),1)'.format(
                signal_path, shot_num)).data()
            ydata = c.get('_sig=dim_of(jet("{}/",{}),0)'.format(
                signal_path, shot_num)).data()
        else:
            time = c.get('_sig=dim_of(jet("{}/",{}))'.format(
                signal_path, shot_num)).data()
        found = True
    except Exception as e:
        g.print_unique(e)
        sys.stdout.flush()
        # pass
    return time, data, ydata, found


def fetch_nstx_data(signal_path, shot_num, c):
    tree, tag = get_tree_and_tag(signal_path)
    c.openTree(tree, shot_num)
    data = c.get(tag).data()
    time = c.get('dim_of(' + tag + ')').data()
    found = True
    return time, data, None, found


d3d = Machine("d3d", "atlas.gat.com", fetch_d3d_data, max_cores=32,
              current_threshold=2e-1)
jet = Machine("jet", "mdsplus.jet.efda.org", fetch_jet_data, max_cores=8,
              current_threshold=1e5)
nstx = Machine("nstx", "skylark.pppl.gov:8501::", fetch_nstx_data, max_cores=8)

all_machines = [d3d, jet]

profile_num_channels = 64

# The "data_avail_tolerances" parameter in Signal class initializer relaxes
# the cutoff for the signal around the defined t_disrupt (provided in the
# disruptive shot list). The latter definition (based on current quench) may
# vary depending on who supplied the shot list and computed t_disrupt, since
# quench may last for O(10 ms). E.g. C. Rea may have taken t_disrupt = midpoint
# of start and end of quench for later D3D shots after 2016 in
# d3d_disrupt_since_2016.txt. Whereas J. Barr, and semi-/automatic methods for
# calculating t_disrupt may use t_disrupt = start of current quench.

# Early-terminating signals will be implicitly padded with zeros when t_disrupt
# still falls within the tolerance (see shots.py,
# Shot.get_signals_and_times_from_file). Even tols > 30 ms are fine (do not
# violate causality), but the ML method may start to base predictions on the
# disappearance of signals.

# "t" subscripted variants of signal variables increase the tolernace to 29 ms
# on D3D, the maximum value possible without violating causality for the min
# T_warn=30 ms. This is important for the signals of newer shots in
# d3d_disrupt_since_2016.txt; many of them would cause [omit] of entire shot
# because their values end shortly before t_disrupt (poss. due to different
# t_disrupt label calculation).

# See conf_parser.py dataset definitions of d3d_data_max_tol, d3d_data_garbage
# which use these signal variants.

# For non-t-subscripted profile signals (and q95), a positive tolerance of
# 20ms on D3D (and 30-50ms on JET) is used to account for the causal shifting
# of the delayed "real-time processing".

# List ---> individual tolerance for each machine when signal definitions are
# shared in cross-machine studies.

# ZIPFIT comes from actual measurements
# jet and d3d:
etemp_profile = ProfileSignal(
    "Electron temperature profile",
    ["ppf/hrts/te", "ZIPFIT01/PROFILES.ETEMPFIT"], [jet, d3d],
    mapping_paths=["ppf/hrts/rho", None], causal_shifts=[0, 10],
    mapping_range=(0, 1), num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.02])
edens_profile = ProfileSignal(
    "Electron density profile",
    ["ppf/hrts/ne", "ZIPFIT01/PROFILES.EDENSFIT"], [jet, d3d],
    mapping_paths=["ppf/hrts/rho", None], causal_shifts=[0, 10],
    mapping_range=(0, 1), num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.02])

etemp_profilet = ProfileSignal(
    "Electron temperature profile tol",
    ["ppf/hrts/te", "ZIPFIT01/PROFILES.ETEMPFIT"], [jet, d3d],
    mapping_paths=["ppf/hrts/rho", None], causal_shifts=[0, 10],
    mapping_range=(0, 1), num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.029])
edens_profilet = ProfileSignal(
    "Electron density profile tol",
    ["ppf/hrts/ne", "ZIPFIT01/PROFILES.EDENSFIT"], [jet, d3d],
    mapping_paths=["ppf/hrts/rho", None], causal_shifts=[0, 10],
    mapping_range=(0, 1), num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.029])
# d3d only:
# etemp_profile = ProfileSignal(
#     "Electron temperature profile", ["ZIPFIT01/PROFILES.ETEMPFIT"], [d3d],
#     mapping_paths=[None], causal_shifts=[10], mapping_range=(0, 1),
#     num_channels=profile_num_channels, data_avail_tolerances=[0.02])
# edens_profile = ProfileSignal(
#     "Electron density profile", ["ZIPFIT01/PROFILES.EDENSFIT"], [d3d],
#     mapping_paths=[None], causal_shifts=[10], mapping_range=(0, 1),
#     num_channels=profile_num_channels, data_avail_tolerances=[0.02])

itemp_profile = ProfileSignal(
    "Ion temperature profile", ["ZIPFIT01/PROFILES.ITEMPFIT"], [d3d],
    causal_shifts=[10], mapping_range=(0, 1),
    num_channels=profile_num_channels, data_avail_tolerances=[0.02])
zdens_profile = ProfileSignal(
    "Impurity density profile", ["ZIPFIT01/PROFILES.ZDENSFIT"], [d3d],
    causal_shifts=[10], mapping_range=(0, 1),
    num_channels=profile_num_channels, data_avail_tolerances=[0.02])
trot_profile = ProfileSignal(
    "Rotation profile", ["ZIPFIT01/PROFILES.TROTFIT"], [d3d],
    causal_shifts=[10], mapping_range=(0, 1),
    num_channels=profile_num_channels, data_avail_tolerances=[0.02])
# note, thermal pressure doesn't include fast ions
pthm_profile = ProfileSignal(
    "Thermal pressure profile", ["ZIPFIT01/PROFILES.PTHMFIT"], [d3d],
    causal_shifts=[10], mapping_range=(0, 1),
    num_channels=profile_num_channels, data_avail_tolerances=[0.02])
neut_profile = ProfileSignal(
    "Neutrals profile", ["ZIPFIT01/PROFILES.NEUTFIT"], [d3d],
    causal_shifts=[10], mapping_range=(0, 1),
    num_channels=profile_num_channels, data_avail_tolerances=[0.02])
# compare the following profile to just q95 0D signal
q_profile = ProfileSignal(
    "Q profile", ["ZIPFIT01/PROFILES.BOOTSTRAP.QRHO"], [d3d],
    causal_shifts=[10], mapping_range=(0, 1),
    num_channels=profile_num_channels, data_avail_tolerances=[0.02])
bootstrap_current_profile = ProfileSignal(
    "Rotation profile", ["ZIPFIT01/PROFILES.BOOTSTRAP.JBS_SAUTER"], [d3d],
    causal_shifts=[10], mapping_range=(0, 1),
    num_channels=profile_num_channels, data_avail_tolerances=[0.02])

# equilibrium_image = 2DSignal(
#     "2D Magnetic Equilibrium", ["EFIT01/RESULTS.GEQDSK.PSIRZ"], [d3d],
#     causal_shifts=[10], mapping_range=(0, 1),
#     num_channels=profile_num_channels, data_avail_tolerances=[0.02])

# EFIT is the solution to the inverse problem from external magnetic
# measurements

# pressure might be unphysical since it is not constrained by measurements,
# only the EFIT which does not know about density and temperature

# pressure_profile = ProfileSignal(
#     "Pressure profile", ["EFIT01/RESULTS.GEQDSK.PRES"], [d3d],
#     causal_shifts=[10], mapping_range=(0, 1),
#     num_channels=profile_num_channels, data_avail_tolerances=[0.02])

q_psi_profile = ProfileSignal(
    "Q(psi) profile", ["EFIT01/RESULTS.GEQDSK.QPSI"], [d3d],
    causal_shifts=[10], mapping_range=(0, 1),
    num_channels=profile_num_channels, data_avail_tolerances=[0.02])

# epress_profile_spatial = ProfileSignal(
#     "Electron pressure profile", ["ppf/hrts/pe/"], [jet], causal_shifts=[25],
#     mapping_range=(2, 4), num_channels=profile_num_channels)

etemp_profile_spatial = ProfileSignal(
    "Electron temperature profile", ["ppf/hrts/te"], [jet],
    causal_shifts=[0], mapping_range=(2, 4),
    num_channels=profile_num_channels, data_avail_tolerances=[0.05])
edens_profile_spatial = ProfileSignal(
    "Electron density profile", ["ppf/hrts/ne"], [jet],
    causal_shifts=[0], mapping_range=(2, 4),
    num_channels=profile_num_channels, data_avail_tolerances=[0.05])
rho_profile_spatial = ProfileSignal(
    "Rho at spatial positions", ["ppf/hrts/rho"], [jet],
    causal_shifts=[0], mapping_range=(2, 4),
    num_channels=profile_num_channels, data_avail_tolerances=[0.05])

etemp = Signal("electron temperature", ["ppf/hrtx/te0"],
               [jet], causal_shifts=[25], data_avail_tolerances=[0.05])
# epress = Signal("electron pressure", ["ppf/hrtx/pe0/"], [jet],
#                 causal_shifts=[25])

q95 = Signal(
    "q95 safety factor", ['ppf/efit/q95', "EFIT01/RESULTS.AEQDSK.Q95"],
    [jet, d3d], causal_shifts=[15, 10], normalize=False,
    data_avail_tolerances=[0.03, 0.02])
q95t = Signal(
    "q95 safety factor tol", ['ppf/efit/q95', "EFIT01/RESULTS.AEQDSK.Q95"],
    [jet, d3d], causal_shifts=[15, 10], normalize=False,
    data_avail_tolerances=[0.03, 0.029])

# "d3d/ipsip" was used before, ipspr15V seems to be available for a
# superset of shots.
ip = Signal("plasma current", ["jpf/da/c2-ipla", "ipspr15V"],
            [jet, d3d], is_ip=True)

ipt = Signal("plasma current tol", ["jpf/da/c2-ipla", "ipspr15V"],
             [jet, d3d], is_ip=True, data_avail_tolerances=[0.029, 0.029])
iptarget = Signal("plasma current target", ["ipsiptargt"], [d3d])
iptargett = Signal("plasma current target tol", ["ipsiptargt"], [d3d],
                   data_avail_tolerances=[0.029])
iperr = Signal("plasma current error", ["ipeecoil"], [d3d])
iperrt = Signal("plasma current error tol", ["ipeecoil"], [d3d],
                data_avail_tolerances=[0.029])

li = Signal("internal inductance", ["jpf/gs/bl-li<s", "efsli"], [jet, d3d])
lit = Signal("internal inductance tol", ["jpf/gs/bl-li<s", "efsli"],
             [jet, d3d], data_avail_tolerances=[0.029, 0.029])
lm = Signal("Locked mode amplitude", ['jpf/da/c2-loca', 'dusbradial'],
            [jet, d3d])
lmt = Signal("Locked mode amplitude tol", ['jpf/da/c2-loca', 'dusbradial'],
             [jet, d3d], data_avail_tolerances=[0.029, 0.029])
dens = Signal("Plasma density", ['jpf/df/g1r-lid:003', 'dssdenest'],
              [jet, d3d], is_strictly_positive=True)
denst = Signal("Plasma density tol", ['jpf/df/g1r-lid:003', 'dssdenest'],
               [jet, d3d], is_strictly_positive=True,
               data_avail_tolerances=[0.029, 0.029])
energy = Signal("stored energy", ['jpf/gs/bl-wmhd<s', 'efswmhd'],
                [jet, d3d])
energyt = Signal("stored energy tol", ['jpf/gs/bl-wmhd<s', 'efswmhd'],
                 [jet, d3d], data_avail_tolerances=[0.029, 0.029])
# Total beam input power
pin = Signal("Input Power (beam for d3d)", ['jpf/gs/bl-ptot<s', 'bmspinj'],
             [jet, d3d])
pint = Signal("Input Power (beam for d3d) tol",
              ['jpf/gs/bl-ptot<s', 'bmspinj'],
              [jet, d3d], data_avail_tolerances=[0.029, 0.029])

pradtot = Signal("Radiated Power", ['jpf/db/b5r-ptot>out'], [jet])
pradtott = Signal("Radiated Power tol", ['jpf/db/b5r-ptot>out'], [jet],
                  data_avail_tolerances=[0.029])
# pradtot = Signal("Radiated Power", ['jpf/db/b5r-ptot>out',
# r'\prad_tot'], [jet,d3d])
# pradcore = ChannelSignal("Radiated Power Core", [r'\bol_l15_p']
# ,[d3d])
# pradedge = ChannelSignal("Radiated Power Edge", [r'\bol_l03_p'],
# [d3d])
pradcore = ChannelSignal("Radiated Power Core",
                         ['ppf/bolo/kb5h/channel14', r'\bol_l15_p'],
                         [jet, d3d])
pradedge = ChannelSignal("Radiated Power Edge",
                         ['ppf/bolo/kb5h/channel10', r'\bol_l03_p'],
                         [jet, d3d])
pradcoret = ChannelSignal("Radiated Power Core tol",
                          ['ppf/bolo/kb5h/channel14', r'\bol_l15_p'],
                          [jet, d3d], data_avail_tolerances=[0.029, 0.029])
pradedget = ChannelSignal("Radiated Power Edge tol",
                          ['ppf/bolo/kb5h/channel10', r'\bol_l03_p'],
                          [jet, d3d], data_avail_tolerances=[0.029, 0.029])
# pechin = Signal("ECH input power, not always on", ['pcechpwrf'], [d3d])
pechin = Signal("ECH input power, not always on",
                ['RF/ECH.TOTAL.ECHPWRC'], [d3d])
pechint = Signal("ECH input power, not always on tol",
                 ['RF/ECH.TOTAL.ECHPWRC'], [d3d],
                 data_avail_tolerances=[0.029])

# betan = Signal("Normalized Beta", ['jpf/gs/bl-bndia<s', 'efsbetan'],
# [jet, d3d])
betan = Signal("Normalized Beta", ['efsbetan'], [d3d])
betant = Signal("Normalized Beta tol", ['efsbetan'], [d3d],
                data_avail_tolerances=[0.029])
energydt = Signal(
    "stored energy time derivative", ['jpf/gs/bl-fdwdt<s'], [jet])

torquein = Signal("Input Beam Torque", ['bmstinj'], [d3d])
torqueint = Signal("Input Beam Torque tol", ['bmstinj'], [d3d],
                   data_avail_tolerances=[0.029])
tmamp1 = Signal("Tearing Mode amplitude (rotating 2/1)", ['nssampn1l'],
                [d3d])
tmamp2 = Signal("Tearing Mode amplitude (rotating 3/2)", ['nssampn2l'],
                [d3d])
tmfreq1 = Signal("Tearing Mode frequency (rotating 2/1)", ['nssfrqn1l'],
                 [d3d])
tmfreq2 = Signal("Tearing Mode frequency (rotating 3/2)", ['nssfrqn2l'],
                 [d3d])
ipdirect = Signal("plasma current direction", ["iptdirect"], [d3d])
ipdirectt = Signal("plasma current direction tol", ["iptdirect"], [d3d],
                   data_avail_tolerances=[0.029])

# for downloading, modify this to preprocess shots with only a subset of
# signals. This may produce more shots
# since only those shots that contain all_signals contained here are used.

# Restricted subset to those signals that are present for most shots. The
# idea is to remove signals that cause many shots to be dropped from the
# dataset.

all_signals = {
    'q95': q95, 'li': li, 'ip': ip, 'betan': betan, 'energy': energy, 'lm': lm,
    'dens': dens, 'pradcore': pradcore,
    'pradedge': pradedge, 'pradtot': pradtot, 'pin': pin,
    'torquein': torquein,
    'energydt': energydt, 'ipdirect': ipdirect, 'iptarget': iptarget,
    'iperr': iperr,
    # 'tmamp1':tmamp1, 'tmamp2':tmamp2, 'tmfreq1':tmfreq1, 'tmfreq2':tmfreq2,
    # 'pechin':pechin,
    # 'rho_profile_spatial':rho_profile_spatial, 'etemp':etemp,
    # -----
    # TODO(KGF): replace this hacky workaround
    # IMPORTANT: must comment-out the following line when preprocessing for
    # training on JET CW and testing on JET ILW (FRNN 0D).
    # Otherwise 1K+ CW shots are excluded due to missing profile data
    'etemp_profile': etemp_profile, 'edens_profile': edens_profile,
    # 'itemp_profile':itemp_profile, 'zdens_profile':zdens_profile,
    # 'trot_profile':trot_profile, 'pthm_profile':pthm_profile,
    # 'neut_profile':neut_profile, 'q_profile':q_profile,
    # 'bootstrap_current_profile':bootstrap_current_profile,
    # 'q_psi_profile':q_psi_profile}
}

all_signals_max_tol = {
    'q95t': q95t, 'lit': lit, 'ipt': ipt, 'betant': betant,
    'energyt': energyt, 'lmt': lmt,
    'denst': denst, 'pradcoret': pradcoret,
    'pradedget': pradedget, 'pint': pint,
    'torqueint': torqueint,
    'ipdirectt': ipdirectt, 'iptargett': iptargett,
    'iperrt': iperrt,
    'etemp_profilet': etemp_profilet, 'edens_profilet': edens_profilet,
}

# for actual data analysis, use:
# all_signals_restricted = [q95, li, ip, energy, lm, dens, pradcore, pradtot,
# pin, etemp_profile, edens_profile]
all_signals_restricted = all_signals

g.print_unique('All signals (determines which signals are downloaded'
               ' & preprocessed):')
g.print_unique(all_signals.values())


fully_defined_signals = {
    sig_name: sig for (sig_name, sig) in all_signals_restricted.items() if (
        sig.is_defined_on_machines(all_machines))
}
fully_defined_signals_0D = {
    sig_name: sig for (sig_name, sig) in all_signals_restricted.items() if (
        sig.is_defined_on_machines(all_machines) and sig.num_channels == 1)
}
fully_defined_signals_1D = {
    sig_name: sig for (sig_name, sig) in all_signals_restricted.items() if (
        sig.is_defined_on_machines(all_machines) and sig.num_channels > 1)
}
d3d_signals = {
    sig_name: sig for (sig_name, sig) in all_signals_restricted.items() if (
        sig.is_defined_on_machine(d3d))
}
d3d_signals_max_tol = {
    sig_name: sig for (sig_name, sig) in all_signals_max_tol.items() if (
        sig.is_defined_on_machine(d3d))
}
d3d_signals_0D = {
    sig_name: sig for (sig_name, sig) in all_signals_restricted.items() if (
        (sig.is_defined_on_machine(d3d) and sig.num_channels == 1))
}
d3d_signals_1D = {
    sig_name: sig for (sig_name, sig) in all_signals_restricted.items() if (
        (sig.is_defined_on_machine(d3d) and sig.num_channels > 1))
}

jet_signals = {
    sig_name: sig for (sig_name, sig) in all_signals_restricted.items() if (
        sig.is_defined_on_machine(jet))
}
jet_signals_0D = {
    sig_name: sig for (sig_name, sig) in all_signals_restricted.items() if (
        (sig.is_defined_on_machine(jet) and sig.num_channels == 1))
}
jet_signals_1D = {
    sig_name: sig for (sig_name, sig) in all_signals_restricted.items() if (
        (sig.is_defined_on_machine(jet) and sig.num_channels > 1))
}

# ['pcechpwrf'] #Total ECH Power Not always on!
# ## 0D EFIT signals ###
# signal_paths += ['EFIT02/RESULTS.AEQDSK.Q95']

# ## 1D EFIT signals ###
# the other signals give more reliable data
# signal_paths += [
#  # Note, the following signals are uniformly mapped over time
# 'AOT/EQU.t_e', # electron temperature profile vs rho
# 'AOT/EQU.dens_e'] # electron density profile vs rho


# [[' $I_{plasma}$ [A]'],
# [' Mode L. A. [A]'],
# [' $P_{radiated}$ [W]'],
# [' $P_{radiated}$ [W]'],
# [' $\rho_{plasma}$ [m^-2]'],
# [' $L_{plasma,internal}$'],
# ['$\frac{d}{dt} E_{D}$ [W]'],
# [' $P_{input}$ [W]'],
# ['$E_{D}$'],
# ppf signal labels
# ['ECE unit?']]
