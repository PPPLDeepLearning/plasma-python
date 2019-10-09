from plasma.utils.downloading import download_all_shot_numbers
from plasma.conf import conf


prepath = '/p/datad2/'  # '/cscratch/share/frnn/'
shot_numbers_path = 'shot_lists/'
save_path = 'signal_data_new/'
# d3d, jet  # should match with data set from conf.yaml
machine = conf['paths']['all_machines'][0]
signals = conf['paths']['all_signals']  # jet_signals, d3d_signals
print('using signals: ')
print(signals)

# shot_list_files = plasma.conf.jet_full
# shot_list_files = plasma.conf.d3d_full
shot_list_files = conf['paths']['shot_files'][0]  # plasma.conf.d3d_100

download_all_shot_numbers(prepath, save_path, shot_list_files, signals)
