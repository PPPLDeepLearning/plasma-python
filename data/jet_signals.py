# JET signal hierarchy
# ------------------------------------------------------------------------
# User only needs to look at 1st and last sections
#    - conf.py only needs to import signals_dirs and signals_masks
#    - get_mdsplus_data.py only needs signals_dirs and download_masks
#    - performance_analysis_utils.py needs :
#         - signals_dirs, plot_masks, ppf_labels, jpf_labels
# ------------------------------------------------------------------------
################
# Signal names #
################
# This section contains all the exact JET signal strings and their
# groupings by type and dimensionality.
# User should not touch this. Use for reference

#############################
#     0D current signals    #
#############################
da_current = ['c2-ipla']  # Plasma Current [A]
da_lock = ['c2-loca']  # Mode Lock Amplitude [A]

############################
#     Radiation signals    #
############################
# 0D
db_out = ['b5r-ptot>out']  # Radiated Power [W]
# 1D vertical signals, don't use signal 16 and 23
db = []
db += ['b5vr-pbol:{:03d}'.format(i)
       for i in range(1, 28) if (i != 16 and i != 23)]
# 1D horizontal signals
db += ['b5hr-pbol:{:03d}'.format(i) for i in range(1, 24)]

####################################
#     1D density signals [m^-2]    #
####################################
df = []
# 4 vertical channels and 4 horizontal channels, dont use signal 1
df += ['g1r-lid:{:03d}'.format(i) for i in range(2, 9)]

###########################
#     0D signals et al    #
###########################
gs_inductance = ['bl-li<s']  # Plasma Internal Inductance
gs_fdwdt = ['bl-fdwdt<s']  # Stored Diamagnetic Energy (time derivative)
gs_power_in = ['bl-ptot<s']  # Total input power [W]
gs_wmhd = ['bl-wmhd<s']  # Stored Diamagnetic Energy (total)
gs_gwdens = ['bl-gwdens<s']  # Greenwald density
gs_torb0 = ['bl-torb0<s']  # Toroidal field on axis
gs_minrad = ['bl-minrad<s']  # Minor radius

###################################
#     ECE temperature profiles    #
###################################
kk3 = []
# Temperature of channel i vs time
kk3 += ['te{:02d}'.format(i) for i in range(1, 97)]
# Radial position of channel i vs time
kk3 += ['ra{:02d}'.format(i) for i in range(1, 97)]
# Radial position of channel i mapped onto midplane vs time
kk3 += ['rc{:02d}'.format(i) for i in range(1, 97)]
# General information (PPF)
kk3 += ['gen']

###################################
# Signal subsystems and groupings #
###################################
# Signal groupings by type/dimensionality
jpf = [da_current,
       da_lock,
       db_out,
       db,
       df,
       gs_inductance,
       gs_fdwdt,
       gs_power_in,
       gs_wmhd,
       gs_gwdens,
       gs_torb0,
       gs_minrad]

ppf = [kk3]

# Manually write strings of actual JET subsystems
jpf_str = ['da',  # magnetic diagnostic, essential subs
           'da',
           'db',  # diagnostic subsys, general pulse radiation
           'db',
           'df',  # density diagnostic, essential subsys
           'gs',  # general services, essential subsys
           'gs',
           'gs',
           'gs',
           'gs',
           'gs',
           'gs']

ppf_str = ['kk3']
subsys_str = [jpf_str, ppf_str]

#################
# Signal types  #
#################
signal_type = [jpf, ppf]
signal_type_str = ['jpf', 'ppf']

#########################################
# Build the full paths from the strings #
#########################################

signals_dirs = []
for i in range(0, len(signal_type)):  # jpf or ppf
    for j in range(0, len(signal_type[i])):  # subsystem/grouping
        signal_group = []
        for k in range(0, len(signal_type[i][j])):  # signal name
            signal_group += [signal_type_str[i] + "/"
                             + subsys_str[i][j] + "/" + signal_type[i][j][k]]
        signals_dirs += [signal_group]
# Print out 2D hierarchy of the signals_dirs
# for i, group in enumerate(signals_dirs):
#     for j, signal in enumerate(group):
#         print(i,j,signal)

##################################################
#             USER SELECTIONS                    #
##################################################


##################################
# Select signals for downloading #
##################################

# Default pass to get_mdsplus_data.py: download all above signals
download_masks = [[True]*len(sig_list) for sig_list in signals_dirs]
# download all radiation signals
download_masks[3] = [False]*len(signals_dirs[3])
# enable/disable ppf/kk3/gen (only one column)
download_masks[-1][-1] = [False]

#######################################
# Select signals for training/testing #
#######################################

# Default pass to conf.py: train with all above signals, minus gs_fdwdt
# signals_masks = [[True]*len(sig_list) for sig_list in signals_dirs]
signals_masks = [[False]*len(sig_list) for sig_list in signals_dirs]
# jpf = [da_current,
#        da_lock,
#        db_out,
#        db,
#        df,
#        gs_inductance,
#        gs_fdwdt,
#        gs_power_in,
#        gs_wmhd,
#        gs_gwdens,
#        gs_torb0,
#        gs_minrad]


# signals_masks[8] = [True] # total diamagnetic energy
# signals_masks[7] = [False] # total input power
signals_masks[6] = [False]  # time derivative of diamagnetic energy
# signals_masks[5] = [True] # inductance
# signals_masks[4] = [False]*len(signals_dirs[4]) #density
# signals_masks[4][1] = True #central denisty #this was automatically excluded!
# signals_masks[3] = [True]*len(signals_dirs[3]) #radiation, vertical and
# horizontal channels
signals_masks[2] = [False]  # wout
signals_masks[1] = [False]  # locked mode
# need to turn the current on for thresholding signal start/end
signals_masks[0] = [True]

# One way for user to disable a signal_group: know the exact index
# signals_masks[6] = [False] # Disable time-derivative of stored diamagnetic
#                              energy
# signals_masks[3] = [False]*len(signals_dirs[3]) #Disable all radiation
# signals

# Another way for the user to disable individual signals within a group:
# Disable 'radial position of channel i vs time' of ECE for all 96 channels
for i, group in enumerate(signals_dirs):
    for j, signal in enumerate(group):
        if 'ra' in signal:
            signals_masks[i][j] = False


# num_signals = sum([group.count(True)
#  for i, group in enumerate(jet_signals.signals_masks)]
###########################################
# Select signals for performance analysis #
###########################################

# User selects these by signal name
plot_masks = [[False]*len(sig_list) for sig_list in signals_dirs]
# Default: 8 golden 0D signals, and all density channels
plot_masks[0][0] = True

# plot_masks[1][0] = True
# plot_masks[2][0] = True
# plot_masks[4][1] = True
# plot_masks[4] = [True]*len(signals_dirs[4]) #density
# plot_masks[5][0] = True
plot_masks[6][0] = True
# plot_masks[7][0] = True
# plot_masks[8][0] = True

# LaTeX strings for performance analysis, sorted in lists by signal_group
group_labels = [[r' $I_{plasma}$ [A]'],
                [r' Mode L. A. [A]'],
                [r' $P_{radiated}$ [W]'],  # 0d radiation, db/
                [r' $P_{radiated}$ [W]'],  # 1d radiation, db/
                [r' $\rho_{plasma}$ [m^-2]'],
                [r' $L_{plasma,internal}$'],
                [r'$\frac{d}{dt} E_{D}$ [W]'],
                [r' $P_{input}$ [W]'],
                [r'$E_{D}$'],
                # ppf signal labels
                [r'ECE unit?']]
