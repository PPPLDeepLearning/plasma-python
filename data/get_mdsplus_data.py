from __future__ import print_function
'''
http://www.mdsplus.org/index.php?title=Documentation:Tutorial:RemoteAccess&open=76203664636339686324830207&page=Documentation%2FThe+MDSplus+tutorial%2FRemote+data+access+in+MDSplus
http://piscope.psfc.mit.edu/index.php/MDSplus_%26_python#Simple_example_of_reading_MDSplus_data
http://www.mdsplus.org/documentation/beginners/expressions.shtml
http://www.mdsplus.org/index.php?title=Documentation:Tutorial:MdsObjects&open=76203664636339686324830207&page=Documentation%2FThe+MDSplus+tutorial%2FThe+Object+Oriented+interface+of+MDSPlus
'''

'''TODO
- mapping to flux surfaces: its not always [0,1]!
- handling of 1D signals during preprocessing & normalization
- handling of 1D signals for feeding into RNN (convolutional layers)
- handling of missing data in shots?
- TEST
'''
from MDSplus import *
#from pylab import *
import numpy as np
import sys
import multiprocessing as mp
from functools import partial
import Queue
import os
import errno

# import gadata

from plasma.primitives.shots import ShotList

from signal_names import *

#print("Importing numpy version"+np.__version__)

def create_missing_value_filler():
	time = np.linspace(0,100,100)
	vals = np.zeros_like(time)
	return time,vals

def mkdirdepth(filename):
	folder=os.path.dirname(filename)
	if not os.path.exists(folder):
		os.makedirs(folder)

def format_save_path(prepath,signal_path,shot_num):
	return prepath + signal_path  + '/{}.txt'.format(shot_num)


def save_shot_signals(shot_num_queue,complete_queue,c,signals,save_prepath,machine):
	missing_values = 0
	# if machine == 'd3d':
	# 	reload(gadata) #reloads Gadata object with connection
	while True:
		try:
			shot_num = shot_num_queue.get(False)
		except Queue.Empty:
			break
		shot_complete = True
		for signal in signals:
			signal_path = signal.get_path(machine)
			save_path_full = format_save_path(save_prepath,signal_path,shot_num)
			success = False
			if os.path.isfile(save_path_full):
				print('-',end='')
			else:
				try:
					try:
						time,data,mapping,success = machine.fetch_data(signal,shot_num,c) 
					except:
						missing_values += 1
						print('Signal {}, shot {} missing. Filling with zeros'.format(signal_path,shot_num))
						time,data = create_missing_value_filler()

					data_two_column = np.vstack((np.atleast_2d(time),np.atleast_2d(data))).transpose()
					try: #can lead to race condition
						mkdirdepth(save_path_full)
					except OSError, e:
					    if e.errno == errno.EEXIST:
					        # File exists, and it's a directory, another process beat us to creating this dir, that's OK.
					        pass
					    else:
					        # Our target dir exists as a file, or different error, reraise the error!
					        raise
					np.savetxt(save_path_full,data_two_column,fmt = '%.5e')#fmt = '%f %f')
					print('.',end='')
				except:
					print('Could not save shot {}, signal {}'.format(shot_num,signal_path))
					print('Warning: Incomplete!!!')
					raise
			sys.stdout.flush()
			if not success:
				missing_values += 1
				shot_complete = False
		#only add shot to list if it was complete
		if shot_complete:
			print('saved shot {}'.format(shot_num))
			complete_queue.put(shot_num)
		else:
			print('shot {} not complete. removing from list.'.format(shot_num))
	print('Finished with {} missing values total'.format(missing_values))



prepath = '/cscratch/share/frnn/'
shot_numbers_path = 'shot_lists/'
save_path = 'signal_data/'
machine = d3d
signals = all_signals




save_prepath = prepath+save_path + '/' + machine + '/'

shot_numbers,_ = ShotList.get_multiple_shots_and_disruption_times(prepath + shot_numbers_path,shot_numbers_files)


fn = partial(save_shot,signals=signals,save_prepath=save_prepath,machine=machine)
num_cores = min(mp.cpu_count(),32) #can only handle 8 connections at once :(
queue = mp.Queue()
complete_queue = mp.Queue()
for shot_num in shot_numbers:
	queue.put(shot_num)
connections = [Connection(server_path) for _ in range(num_cores)]
processes = [mp.Process(target=fn,args=(queue,connections[i])) for i in range(num_cores)]

print('running in parallel on {} processes'.format(num_cores))
start_time = time.time()

for p in processes:
	p.start()
for p in processes:
	p.join()

complete_shot_numbers = list(complete_queue)
print('Finished downloading {} ({} complete) shots in {} seconds'.format(len(shot_numbers),len(complete_shot_numbers),time.time()-start_time))


# c = Connection(server_path)

# pool = mp.Pool()
# for shot_num in shot_numbers:
# # 	save_shot(shot_num,signal_paths,save_prepath,machine,c)
# for (i,_) in enumerate(pool.imap_unordered(fn,shot_numbers)):
#     print('{}/{}'.format(i,len(shot_numbers)))

# pool.close()
# pool.join()





# def save_shot(shot_num_queue,c,signal_paths,save_prepath,machine):
# 	missing_values = 0
# 	# if machine == 'd3d':
# 	# 	reload(gadata) #reloads Gadata object with connection
# 	while True:
# 		try:
# 			shot_num = shot_num_queue.get(False)
# 		except Queue.Empty:
# 			break
# 		for signal_path in signal_paths:
# 			save_path_full = format_save_path(save_prepath,signal_path,shot_num)
# 			if os.path.isfile(save_path_full):
# 				print('-',end='')
# 			else:
# 			    try:
# 					if machine == 'nstx':
# 						tree,tag = get_tree_and_tag(signal_path)
# 						c.openTree(tree,shot_num)
# 						data = c.get(tag).data()
# 						time = c.get('dim_of('+tag+')').data()
# 					elif machine == 'jet':
# 						try:
# 							data = c.get('_sig=jet("{}/",{})'.format(signal_path,shot_num)).data()
# 							time = c.get('_sig=dim_of(jet("{}/",{}))'.format(signal_path,shot_num)).data()
# 						except:
# 							missing_values += 1
# 							print('Signal {}, shot {} missing. Filling with zeros'.format(signal_path,shot_num))
# 							time,data = create_missing_value_filler()
# 					elif machine == 'd3d':
# 						tree,tag = get_tree_and_tag_no_backslash(signal_path)
# 						try:
# 							ga1 = gadata.gadata('{}'.format(tag),shot_num,tree=tree,connection=c)
# 							if not ga1.found:
# 								raise
# #							ga1 = gadata.gadata('\\{}'.format(signal_path),shot_num,tree='d3d',connection=c)
# 							data = ga1.zdata
# 							time = ga1.xdata					
# 						except:
# 							missing_values += 1
# 							print('Signal {}, shot {} missing. Filling with zeros'.format(signal_path,shot_num))
# 							time,data = create_missing_value_filler()

# 					data_two_column = np.vstack((np.atleast_2d(time),np.atleast_2d(data))).transpose()
# 					try: #can lead to race condition
# 						mkdirdepth(save_path_full)
# 					except OSError, e:
# 					    if e.errno == errno.EEXIST:
# 					        # File exists, and it's a directory, another process beat us to creating this dir, that's OK.
# 					        pass
# 					    else:
# 					        # Our target dir exists as a file, or different error, reraise the error!
# 					        raise
# 					np.savetxt(save_path_full,data_two_column,fmt = '%.5e')#fmt = '%f %f')
# 					print('.',end='')
# 			    except:
# 					print('Could not save shot {}, signal {}'.format(shot_num,signal_path))
# 					print('Warning: Incomplete!!!')
# 					raise
# 			sys.stdout.flush()
# 		print('saved shot {}'.format(shot_num))
# 	print('Finished with {} missing values total'.format(missing_values))


# if machine == 'nstx':
# 	shot_numbers_files = ['disrupt_nstx.txt'] 
# 	server_path = "skylark.pppl.gov:8501::"
# 	signal_paths = ['engineering/ip1/',
# 	'operations/rwmef_plas_n1_amp_br/',
# 	'efit02/li/',
# 	'activespec/ts_ld/',
# 	'passivespec/bolom_totpwr/',
# 	'nbi/nb_p_inj/',
# 	'efit02/wpdot/']

# elif machine == 'd3d':
# 	shot_numbers_files = ['shotlist_JaysonBarr_clear.txt']
# 	shot_numbers_files += ['shotlist_JaysonBarr_disrupt.txt']
# 	#shot_numbers_files = ['d3d_short_clear.txt']# ,'d3d_clear.txt', 'd3d_disrupt.txt']
# 	server_path = 'atlas.gat.com'
# 	from d3d_signals import signal_paths	
# 	import itertools
# 	signal_paths = list(itertools.chain.from_iterable(signal_paths))
# #	signal_paths = ['PINJ','IP','Q95','DENSITY','WMHD'] #,'PRAD'] #PRAD returns a 2D xdata
# #       Recommended signals from DIII-D
# 	# signal_paths = ['efsli','ipsip','efsbetan','efswmhd','nssampn1l','nssfrqn1l',
# 			# 'nssampn2l','nssfrqn2l',
# 			# 'dusbradial','dssdenest',r'\bol_l15_p',r'\bol_l03_p','bmspinj','bmstinj','pcechpwrf']
# 	# signal_paths = ['d3d/' + path for path in signal_paths]
	


# elif machine == 'jet':
# 	shot_numbers_files = ['CWall_clear.txt','CFC_unint.txt','BeWall_clear.txt','ILW_unint.txt']
# 	server_path = 'mdsplus.jet.efda.org'

# 	#plasma current, locked mode, output power
# 	signal_paths = ['jpf/da/c2-ipla',
# 	'jpf/da/c2-loca',
# 	'jpf/db/b5r-ptot>out']



# 	#internal inductance, time derivative of stored energy, input power, total diamagnetic energy
# 	signal_paths += ['jpf/gs/bl-li<s',
# 	'jpf/gs/bl-fdwdt<s',
# 	'jpf/gs/bl-ptot<s',
# 	'jpf/gs/bl-wmhd<s']



	
# 	#density signals
# 	#4 vertical channels and 4 horizontal channels
# 	signal_paths += ['jpf/df/g1r-lid:{:03d}'.format(i) for i in range(1,9)]



# 	#radiation signals
# 	#vertical signals, don't use signal 16 and 23
# 	signal_paths += ['jpf/db/b5vr-pbol:{:03d}'.format(i) for i in range(1,28) if (i != 16 and i != 23)]
# 	signal_paths += ['jpf/db/b5hr-pbol:{:03d}'.format(i) for i in range(1,24)]


# 	#ece temperature profiles
# 	#temperature of channel i vs time
# 	signal_paths += ['ppf/kk3/te{:02d}'.format(i) for i in range(1,97)]

# 	#radial position of channel i vs time
# 	#signal_paths += ['ppf/kk3/ra{:02d}'.format(i) for i in range(1,97)]

# 	#radial position of channel i mapped onto midplane vs time
# 	signal_paths += ['ppf/kk3/rc{:02d}'.format(i) for i in range(1,97)]







# else:
# 	print('unkown machine. exiting')
# 	exit(1)


