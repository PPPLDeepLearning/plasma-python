from MDSplus import *
import numpy as np
import time
import sys

# class SignalCollection:
# 	"""GA Data Obj"""
# 	def __init__(self,signal_descriptions,signal_paths):
# 		self.signals = []
# 		for i in range(len(signal_paths))
# 			self.signals.append(Signal(signal_descriptions[i],signal_paths[i]))

class Signal:
	def __init__(self,description,paths,machines=['jet'],tex_label=None,causal_shifts=None):
		assert(len(paths) == len(machines))
		self.description = description
		self.paths = paths
		self.machines = machines #on which machines is the signal defined
		if causal_shifts == None:
			causal_shifts = [0 for m in machines]
		self.causal_shifts = causal_shifts #causal shift in ms

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


class Machine:
	def __init__(self,name,server,fetch_data_fn,max_cores = 8):
		self.name = name
		self.server = server
		self.max_cores = max_cores
		self.fetch_data_fn = fetch_data_fn

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


def create_missing_value_filler():
	time = np.linspace(0,100,100)
	vals = np.zeros_like(time)
	return time,vals

def get_tree_and_tag(path):
	spl = path.split('/')
	tree = spl[0]
	tag = '\\' + spl[1]
	return tree,tag

def get_tree_and_tag_no_backslash(path):
	spl = path.split('/')
	tree = spl[0]
	tag = spl[1]
	return tree,tag

def fetch_d3d_data(signal_path,shot,c=None):
	tree,signal = get_tree_and_tag_no_backslash(signal_path)
	if tree == None:
		signal = c.get('findsig("'+signal+'",_fstree)').value
		tree = c.get('_fstree').value 
	if c is None:
		c = MDSplus.Connection('atlas.gat.com')

	# ## Retrieve Data 
	t0 =  time.time()
	found = False
	xdata = [0]
	ydata = None
	data = [0]

	# Retrieve data from MDSplus (thin)
	#first try, retrieve directly from tree andsignal 
	def get_units(str):
		units = c.get('units_of('+str+')').data()
		if units == '' or units == ' ': 
			units = c.get('units('+str+')').data()
		return units

	try:     
		c.openTree(tree,shot)
		data  = c.get('_s = '+signal).data()
		data_units = c.get('units_of(_s)').data()  
		rank = np.ndim(data)	
		if rank > 1:
			xdata = c.get('dim_of(_s,1)').data()
			xunits = get_units('dim_of(_s,1)')
			ydata 	= c.get('dim_of(_s)').data()
	    		yunits = get_units('dim_of(_s)')
		else:
			xdata = c.get('dim_of(_s)').data()
	    		xunits = get_units('dim_of(_s)')
		found = True
		# MDSplus seems to return 2-D arrays transposed.  Change them back.
		if np.ndim(data) == 2: data = np.transpose(data)
		if np.ndim(ydata) == 2: ydata = np.transpose(ydata)
		if np.ndim(xdata) == 2: xdata = np.transpose(xdata)

	except Exception,e:
		#print(e)
		#sys.stdout.flush()
		pass

	# Retrieve data from PTDATA if node not found
	if not found:
		#print("not in full path {}".format(signal))
		data = c.get('_s = ptdata2("'+signal+'",'+str(shot)+')')
		if len(data) != 1:
			xdata = c.get('dim_of(_s)')
			rank = 1
			found = True
	# Retrieve data from Pseudo-pointname if not in ptdata
	if not found:
		#print("not in PTDATA {}".format(signal))
		data = c.get('_s = pseudo("'+signal+'",'+str(shot)+')')
		if len(data) != 1:
			xdata = c.get('dim_of(_s)')
			rank = 1
			found = True
	#this means the signal wasn't found
	if not found:  
		print "   No such signal: %s" % (signal,)
		pass

    # print '   GADATA Retrieval Time : ',time.time() - t0
	return xdata,data,ydata,found


def fetch_jet_data(signal_path,shot_num,c):
	data = c.get('_sig=jet("{}/",{})'.format(signal_path,shot_num)).data()
	time = c.get('_sig=dim_of(jet("{}/",{}))'.format(signal_path,shot_num)).data()
	found = True
	return time,data,None,found

def fetch_nstx_data(signal_path,shot_num,c):
	tree,tag = get_tree_and_tag(signal_path)
	c.openTree(tree,shot_num)
	data = c.get(tag).data()
	time = c.get('dim_of('+tag+')').data()
	found = True
	return time,data,None,found

d3d = Machine("d3d","atlas.gat.com",fetch_d3d_data,max_cores=32)
jet = Machine("jet","mdsplus.jet.efda.org",fetch_jet_data,max_cores=8)
nstx = Machine("nstx","skylark.pppl.gov:8501::",fetch_nstx_data,max_cores=8)


all_machines = [d3d,jet]

etemp_profile = Signal("Electron temperature profile",["ZIPFIT01/PROFILES.ETEMPFIT"],[d3d],causal_shifts=[10])
edens_profile = Signal("Electron density profile",["ZIPFIT01/PROFILES.EDENSFIT"],[d3d],causal_shifts=[10])

q95 = Signal("q95 safety factor",['ppf/efit/q95',"EFIT01/RESULTS.AEQDSK.Q95"],[jet,d3d],causal_shifts=[15,10])

li = Signal("locked mode amplitude",["jpf/gs/bl-li<s","d3d/efsli"],[jet,d3d])
ip = Signal("plasma current",["jpf/da/c2-ipla","d3d/ipsip"],[jet,d3d])
betan = Signal("Normalized Beta",['d3d/efsbetan'],[d3d])
energy = Signal("stored energy",['d3d/efswmhd'],[d3d])
lm = Signal("Locked mode amplitude",['jpf/da/c2-loca','d3d/dusbradial'],[jet,d3d])
dens = Signal("Plasma density",['jpf/df/g1r-lid:003','d3d/dssdenest'],[jet,d3d])

pradcore = Signal("Radiated Power Core",['d3d/'+r'\bol_l15_p'],[d3d])
pradedge = Signal("Radiated Power Edge",['d3d/'+r'\bol_l03_p'],[d3d])
pradtot = Signal("Radiated Power",['jpf/db/b5r-ptot>out'],[jet])
pin = Signal("Input Power (beam for d3d)",['jpf/gs/bl-ptot<s','d3d/bmspinj'],[jet,d3d]) #Total Beam Power
pechin = Signal("ECH input power, not always on",['d3d/pcechpwrf'],[d3d])

torquein = Signal("Input Beam Torque",['d3d/bmstinj'],[d3d]) #Total Beam Power
tmamp1 = Signal("Tearing Mode amplitude (rotating 2/1)", ['d3d/nssampn1l'],[d3d])
tmamp2 = Signal("Tearing Mode amplitude (rotating 3/2)", ['d3d/nssampn2l'],[d3d])
tmfreq1 = Signal("Tearing Mode frequency (rotating 2/1)", ['d3d/nssfrqn1l'],[d3d])
tmfreq2 = Signal("Tearing Mode frequency (rotating 3/2)", ['d3d/nssfrqn2l'],[d3d])


all_signals = [etemp_profile,edens_profile,q95,li,ip,
betan,energy,lm,dens,pradcore,pradedge,pradtot,pin,
torquein,tmamp1,tmamp2,tmfreq1,tmfreq2
#pechin,
]


fully_defined_signals = [sig for sig in all_signals if sig.is_defined_on_machines(all_machines)]
d3d_signals = [sig for sig in all_signals if sig.is_defined_on_machine(d3d)]
jet_signals = [sig for sig in all_signals if sig.is_defined_on_machine(jet)]



#['pcechpwrf'] #Total ECH Power Not always on!
### 0D EFIT signals ###
#signal_paths += ['EFIT02/RESULTS.AEQDSK.Q95']
  
### 1D EFIT signals ###
#the other signals give more reliable data
#signal_paths += [
#'AOT/EQU.t_e', #electron temperature profile vs rho (uniform mapping over time)
#'AOT/EQU.dens_e'] #electron density profile vs rho (uniform mapping over time)






