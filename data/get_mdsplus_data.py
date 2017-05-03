from plasma.utils.downloading import download_all_shot_numbers
from data.signals import *


prepath = '/cscratch/share/frnn/'#'/p/datad2/'
shot_numbers_path = 'shot_lists/'
save_path = 'signal_data/'
machine = d3d#jet#d3d 
signals = d3d_signals#jet_signals#d3d_signals
print('using signals: ')
print(signals)

#nstx
# 	shot_numbers_files = ['disrupt_nstx.txt']  #nstx

#d3d
#shot_numbers_files = ['shotlist_JaysonBarr_clear.txt']
#shot_numbers_files += ['shotlist_JaysonBarr_disrupt.txt']
# 	#shot_numbers_files = ['d3d_short_clear.txt']# ,'d3d_clear.txt', 'd3d_disrupt.txt']
#shot_numbers_files = ['d3d_clear.txt', 'd3d_disrupt.txt']#['d3d_short_clear.txt']# ,'d3d_clear.txt', 'd3d_disrupt.txt'] #data only available after shot 125500
shot_numbers_files = ['d3d_clear_data_avail.txt', 'd3d_disrupt_data_avail.txt']
#jet
#shot_numbers_files = ['CWall_clear.txt','CFC_unint.txt','BeWall_clear.txt','ILW_unint.txt']#jet

download_all_shot_numbers(prepath,save_path,shot_numbers_path,shot_numbers_files,machine)

