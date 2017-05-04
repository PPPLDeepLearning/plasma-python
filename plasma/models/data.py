import numpy as np
import time
import sys
from plasma.utils.processing import get_individual_shot_file

# class SignalCollection:
# 	"""GA Data Obj"""
# 	def __init__(self,signal_descriptions,signal_paths):
# 		self.signals = []
# 		for i in range(len(signal_paths))
# 			self.signals.append(Signal(signal_descriptions[i],signal_paths[i]))

class Signal:
	def __init__(self,description,paths,machines,tex_label=None,causal_shifts=None,is_ip=False):
		assert(len(paths) == len(machines))
		self.description = description
		self.paths = paths
		self.machines = machines #on which machines is the signal defined
		if causal_shifts == None:
			causal_shifts = [0 for m in machines]
		self.causal_shifts = causal_shifts #causal shift in ms
		self.is_ip = is_ip
		self.num_channels = 1

	def is_ip(self):
		return self.is_ip

	def get_file_path(self,prepath,shot):
		dirname = self.get_path(shot.machine)
		return get_individual_shot_file(prepath+dirname + '/',shot)

	def is_valid(self,prepath,shot):
		t,data,exists = self.load_data(prepath,shot)
		return exists 

	def is_saved(self,prepath,shot):
		file_path = self.get_file_path(prepath,shot)
		return os.path.isfile(file_path)

	def load_data(self,prepath,shot):
		if not self.is_saved(prepath,shot):
			print('Signal {}, shot {} was never downloaded'.format(self.description,shot.number))
			return None,None,False

		file_path = self.get_file_path(prepath,shot)
		data = np.loadtxt(file_path)
		t = data[:,0]
		sig = data[:,1:]

		if self.is_ip: #restrict shot to current threshold
			region = np.where(np.abs(sig) >= shot.machine.current_threshold)
			t = t[region]
			sig = sig[region,:]

		#make sure shot is not garbage data
		if (np.max(sig) == 0.0 and np.min(sig) == 0.0) or len(t) <= 1::
			if self.is_ip:
				print('shot {} has no current'.format(shot.number))
			else:
				print('Signal {}, shot {} contains no data'.format(self.description,shot.number))
			return None,None,False

		return t,sig,True

	def resample_signal(t,sig,tmin,tmax,dt):
		order = np.argsort(t)
		t = t[order]
		sig = sig[order,:]
		sig_width = sig.shape[1]
		tt = np.arange(tmin,tmax,dt)
		sig_interp = np.zeros((len(tt),sig_width))
		for i in range(sig_width):
			f = UnivariateSpline(t,sig[:,i],s=0,k=1,ext=0)
			sig_interp[:,i] = f(tt)

		if(np.any(np.isnan(sig_interp))):
			print("signals contains nan")
		if(np.any(t[1:] - t[:-1] <= 0)):
			print("non increasing")
			idx = np.where(t[1:] - t[:-1] <= 0)[0][0]
			print(t[idx-10:idx+10])

		return tt,sig_interp

	def cut_signal(t,sig,tmin,tmax):
		mask = np.logical_and(t >= tmin,  t <= tmax)
		return t[mask],sig[mask,:]

	def cut_and_resample_signal(t,sig,tmin,tmax,dt):
		t,sig = cut_signal(t,sig,tmin,tmax)
		return resample_signal(t,sig,tmin,tmax,dt)

	def is_defined_on_machine(self,machine):
		return machine in self.machines

	def is_defined_on_machines(self,machines):
		return all([m in self.machines for m in machines])

	def get_path(self,machine):
		idx = self.get_idx(machine)
		return self.paths[idx]

	def get_causal_shift(self,machine):
		idx = self.get_idx(machine)
		return self.causal_shifts[idx]

	def get_idx(self,machine):
		assert(machine in self.machines)
		idx = self.machines.index(machine)	
		return idx

	def __eq__(self,other):
		return self.description.__eq__(other.description)

	
	def __ne__(self,other):
		return self.description.__ne__(other.description)
	
	def __hash__(self,other):
		return self.description.__hash__()

	def __str__(self):
		return self.description
	
	def __repr__(self):
		return self.description

class ProfileSignal(Signal):
	def __init__(self,description,paths,machines,tex_label=None,causal_shifts=None,mapping_range=(0,1),num_channels=32):
		super(ProfileSignal, self).__init__(description,paths,machines,tex_label,causal_shifts,is_ip=False)
		self.mapping_range = mapping_range
		self.num_channels = num_channels

	def load_data(self,prepath,shot):
		if not self.is_saved(prepath,shot):
			print('Signal {}, shot {} was never downloaded'.format(self.description,shot.number))
			return None,None,False

		file_path = self.get_file_path(prepath,shot)
		data = np.loadtxt(file_path)
		_ = data[0,0]
		mapping = data[0,1:]
		remapping = np.linspace(self.mapping_range[0],self.mapping_range[1],self.num_channels)
		t = data[1:,0]
		sig = data[1:,1:]
		if len(t) <= 1 or (np.max(sig) == 0.0 and np.min(sig) == 0.0):
			print('Signal {}, shot {} contains no data'.format(self.description,shot.number))
			return None,None,False

		timesteps = len(t)
		sig_interp = np.zeros((timesteps,self.num_channels))
		for i in range(timesteps):
			f = UnivariateSpline(mapping,sig[i,:],s=0,k=1,ext=0)
			sig_interp[i,:] = f(remapping)

		return t,sig_interp,True


class Machine:
	def __init__(self,name,server,fetch_data_fn,max_cores = 8,current_threshold=0):
		self.name = name
		self.server = server
		self.max_cores = max_cores
		self.fetch_data_fn = fetch_data_fn
		self.current_threshold = current_threshold

	def get_connection(self):
		return Connection(server)

	def fetch_data(self,signal,shot_num,c):
		path = signal.get_path(self)
		success = False
		mapping = None
		try:
			time,data,mapping,success = self.fetch_data_fn(path,shot_num,c)
		except Exception,e:
			time,data = create_missing_value_filler()
			print(e)
			sys.stdout.flush()
		time = np.array(time) + signal.get_causal_shift(self)
		return time,np.array(data),mapping,success

	def __eq__(self,other):
		return self.name.__eq__(other.name)

	
	def __ne__(self,other):
		return self.name.__ne__(other.name)
	
	def __hash__(self,other):
		return self.name.__hash__()
	
	def __str__(self):
		return self.name

	def __repr__(self):
		return self.__str__()

def create_missing_value_filler():
	time = np.linspace(0,100,100)
	vals = np.zeros_like(time)
	return time,vals
