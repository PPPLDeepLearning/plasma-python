#JET signal hierarchy
#------------------------------------------------------------------------#
#User only needs to look at 1st and last sections
#    - conf.py only needs to import signals_dirs and signals_masks
#    - get_mdsplus_data.py only needs signals_dirs and download_masks
#    - performance_analysis_utils.py needs :
#         - signals_dirs, plot_masks, ppf_labels, jpf_labels
#------------------------------------------------------------------------#
################
# Signal names #
################
#This section contains all the exact JET signal strings and their
#groupings by type and dimensionality.
#User should not touch this. Use for reference


### 0D signals ###
signal_paths = [
'efsli', #Internal Inductance
'ipsip', #Plasma Current
'efsbetan', #Normalized Beta
'efswmhd', #Stored Energy
'nssampn1l', #Tearing Mode Amplitude (rotating 2/1)
'nssfrqn1l', #Tearing Mode Frequency (rotating 2/1)
'nssampn2l', #Tearing Mode Amplitude (rotating 3/2)
'nssfrqn2l', #Tearing Mode Frequency (rotating 3/2)
'dusbradial', #LM Amplitude
'dssdenest', #Plasma Density
r'\bol_l15_p', #Radiated Power core
r'\bol_l03_p', #Radiated Power Edge
'bmspinj', #Total Beam Power
'bmstinj', #Total Beam Torque
'pcechpwrf'] #Total ECH Power

signal_paths = ['d3d/' + path for path in signal_paths]
  
### 1D EFIT signals ###
signal_paths += [
'AOT/EQU.te', #electron temperature profile vs rho (uniform mapping over time)
'AOT/EQU.dens_e'] #electron density profile vs rho (uniform mapping over time)

#make into list of lists format to be consistent with jet_signals.py
signal_paths = [[path] for path in signal_paths]

#format : 'tree/signal_path' for each path
signals_dirs = signal_paths

 
##################################################
#             USER SELECTIONS                    #
##################################################


##################################
# Select signals for downloading #
##################################

#Default pass to get_mdsplus_data.py: download all above signals
download_masks = [[True]*len(sig_list) for sig_list in signals_dirs]
# download_masks[-1] = [False] # enable/disable temperature profile 
# download_masks[-2] = [False] # enable/disable density profile 

#######################################
# Select signals for training/testing #
#######################################

#Default pass to conf.py: train with all above signals
signals_masks = [[True]*len(sig_list) for sig_list in signals_dirs]
signals_masks[-1] = [False] # enable/disable temperature profile 
signals_masks[-2] = [False] # enable/disable density profile 

#num_signals = sum([group.count(True) for i,group in enumerate(jet_signals.signals_masks)]
###########################################
# Select signals for performance analysis #
###########################################

#User selects these by signal name
plot_masks = [[True]*len(sig_list) for sig_list in signals_dirs]

#LaTeX strings for performance analysis, sorted in lists by signal_group
group_labels = [[r' $I_{plasma}$ [A]'],
              [r' Mode L. A. [A]'],
              [r' $P_{radiated}$ [W]'], #0d radiation, db/
              [r' $P_{radiated}$ [W]'],#1d radiation, db/
              [r' $\rho_{plasma}$ [m^-2]'],
              [r' $L_{plasma,internal}$'],
              [r'$\frac{d}{dt} E_{D}$ [W]'],
              [r' $P_{input}$ [W]'],
              [r'$E_{D}$'],
#ppf signal labels
                [r'ECE unit?']]
