from __future__ import print_function
import numpy as np
import time
import sys

from plasma.models.data import Signal,ProfileSignal,Machine


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
	# if c is None:
		# c = MDSplus.Connection('atlas.gat.com')

	# ## Retrieve Data 
	t0 =  time.time()
	found = False
	xdata = np.array([0])
	ydata = None
	data = np.array([0])

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

	except Exception as e:
		#print(e)
		#sys.stdout.flush()
		pass

	# Retrieve data from PTDATA if node not found
	if not found:
		#print("not in full path {}".format(signal))
		data = c.get('_s = ptdata2("'+signal+'",'+str(shot)+')').data()
		if len(data) != 1:
			xdata = c.get('dim_of(_s)').data()
			rank = 1
			found = True
	# Retrieve data from Pseudo-pointname if not in ptdata
	if not found:
		#print("not in PTDATA {}".format(signal))
		data = c.get('_s = pseudo("'+signal+'",'+str(shot)+')').data()
		if len(data) != 1:
			xdata = c.get('dim_of(_s)').data()
			rank = 1
			found = True
	#this means the signal wasn't found
	if not found:  
		print ("No such signal: {}".format(signal))
		pass

    # print '   GADATA Retrieval Time : ',time.time() - t0
	xdata = xdata*1e-3#time is measued in ms
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





d3d = Machine("d3d","atlas.gat.com",fetch_d3d_data,max_cores=32,current_threshold=1e-1)
jet = Machine("jet","mdsplus.jet.efda.org",fetch_jet_data,max_cores=8,current_threshold=1e5)
nstx = Machine("nstx","skylark.pppl.gov:8501::",fetch_nstx_data,max_cores=8)


all_machines = [d3d,jet]

profile_num_channels = 32
etemp_profile = ProfileSignal("Electron temperature profile",["ZIPFIT01/PROFILES.ETEMPFIT"],[d3d],causal_shifts=[10],mapping_range=(0,1),num_channels=profile_num_channels)
edens_profile = ProfileSignal("Electron density profile",["ZIPFIT01/PROFILES.EDENSFIT"],[d3d],causal_shifts=[10],mapping_range=(0,1),num_channels=profile_num_channels)

q95 = Signal("q95 safety factor",['ppf/efit/q95',"EFIT01/RESULTS.AEQDSK.Q95"],[jet,d3d],causal_shifts=[15,10])

ip = Signal("plasma current",["jpf/da/c2-ipla","d3d/ipsip"],[jet,d3d],is_ip=True)
li = Signal("internal inductance",["jpf/gs/bl-li<s","d3d/efsli"],[jet,d3d])
lm = Signal("Locked mode amplitude",['jpf/da/c2-loca','d3d/dusbradial'],[jet,d3d])
dens = Signal("Plasma density",['jpf/df/g1r-lid:003','d3d/dssdenest'],[jet,d3d])
energy = Signal("stored energy",['jpf/gs/bl-wmhd<s','d3d/efswmhd'],[jet,d3d])
pin = Signal("Input Power (beam for d3d)",['jpf/gs/bl-ptot<s','d3d/bmspinj'],[jet,d3d]) #Total Beam Power

pradtot = Signal("Radiated Power",['jpf/db/b5r-ptot>out'],[jet])
pradcore = Signal("Radiated Power Core",['d3d/'+r'\bol_l15_p'],[d3d])
pradedge = Signal("Radiated Power Edge",['d3d/'+r'\bol_l03_p'],[d3d])
# pechin = Signal("ECH input power, not always on",['d3d/pcechpwrf'],[d3d])
pechin = Signal("ECH input power, not always on",['RF/ECH.TOTAL.ECHPWRC'],[d3d])

betan = Signal("Normalized Beta",['d3d/efsbetan'],[d3d])
energydt = Signal("stored energy time derivative",['jpf/gs/bl-fdwdt<s'],[jet])

torquein = Signal("Input Beam Torque",['d3d/bmstinj'],[d3d]) #Total Beam Power
tmamp1 = Signal("Tearing Mode amplitude (rotating 2/1)", ['d3d/nssampn1l'],[d3d])
tmamp2 = Signal("Tearing Mode amplitude (rotating 3/2)", ['d3d/nssampn2l'],[d3d])
tmfreq1 = Signal("Tearing Mode frequency (rotating 2/1)", ['d3d/nssfrqn1l'],[d3d])
tmfreq2 = Signal("Tearing Mode frequency (rotating 3/2)", ['d3d/nssfrqn2l'],[d3d])
ipdirect = Signal("plasma current direction",["d3d/iptdirect"],[d3d])

#for downloading
all_signals = [q95,li,ip,
betan,energy,lm,dens,pradcore,pradedge,pradtot,pin,
torquein,tmamp1,tmamp2,tmfreq1,tmfreq2,pechin,energydt,ipdirect,etemp_profile,edens_profile,
]

#for actual data analysis
#all_signals_restricted = [q95,li,ip,energy,lm,dens,pradcore,pradtot,pin,etemp_profile,edens_profile]

all_signals_restricted = all_signals

print('all signals:')
print(all_signals)

fully_defined_signals = [sig for sig in all_signals_restricted if sig.is_defined_on_machines(all_machines)]
d3d_signals = [sig for sig in all_signals_restricted if sig.is_defined_on_machine(d3d)]
jet_signals = [sig for sig in all_signals_restricted if sig.is_defined_on_machine(jet)]



#['pcechpwrf'] #Total ECH Power Not always on!
### 0D EFIT signals ###
#signal_paths += ['EFIT02/RESULTS.AEQDSK.Q95']
  
### 1D EFIT signals ###
#the other signals give more reliable data
#signal_paths += [
#'AOT/EQU.t_e', #electron temperature profile vs rho (uniform mapping over time)
#'AOT/EQU.dens_e'] #electron density profile vs rho (uniform mapping over time)


# [[' $I_{plasma}$ [A]'],
#[' Mode L. A. [A]'],
#[' $P_{radiated}$ [W]'],
#[' $P_{radiated}$ [W]'],
#[' $\rho_{plasma}$ [m^-2]'],
#[' $L_{plasma,internal}$'],
#['$\frac{d}{dt} E_{D}$ [W]'],
#[' $P_{input}$ [W]'],
#['$E_{D}$'],
##ppf signal labels
#['ECE unit?']]



