from plasma.utils.downloading import download_all_shot_numbers
from data.signals import *
import plasma.conf


prepath = '/cscratch/share/frnn/'#'/p/datad2/'
shot_numbers_path = 'shot_lists/'
save_path = 'signal_data/'
machine = d3d#jet#d3d 
signals = all_signals#jet_signals#d3d_signals
print('using signals: ')
print(signals)

# shot_list_files = plasma.conf.jet_full
shot_list_files = plasma.conf.d3d_full

download_all_shot_numbers(prepath,save_path,shot_list_files,signals)

