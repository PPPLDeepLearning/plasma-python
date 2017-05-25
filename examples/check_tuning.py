from plasma.models.hyperparameters import CategoricalHyperparam,ContinuousHyperparam,LogContinuousHyperparam
from pprint import pprint
import yaml
import datetime
import uuid
import sys,os,getpass
import shutil
import subprocess as sp
import pandas
import numpy as np

dir_path = "/tigress/jk7/hyperparams/"
if len(sys.argv) <= 1:
    dir_path = dir_path + os.listdir(dir_path)[0] + '/'
    print("using default dir {}".format(dir_path))
else:
    dir_path = sys.argv[1]


class HyperparamExperiment():
    def __init__(self,path,conf_name = "conf.yaml"):
	if not path.endswith('/'):
	    path += '/'
	self.path = path
	self.logs_path = path + "csv_logs/"
	self.raw_logs_path = path[:-1] + ".out"
	self.changed_path = path + "changed_params.out"
	with open(self.path + conf_name, 'r') as yaml_file:
		conf = yaml.load(yaml_file)
	self.name_to_monitor = conf['callbacks']['monitor']
	self.load_data()
	self.get_changed()
	self.get_maximum()
	self.read_raw_logs()

    def __str__(self):
	s = "Experiment:\n"
	s += '-'*20+"\n"
	s += self.changed
	s += '-'*20+"\n"
	s += "Maximum of {} at epoch {}\n".format(*self.get_maximum())
	s += '-'*20+"\n"
	return s

    def load_data(self):
	files = os.listdir(self.logs_path)
	assert(len(files) == 1)
	self.logs_path = self.logs_path + files[0]
	if os.path.getsize(self.logs_path) > 0:
	    dat = pandas.read_csv(self.logs_path)
	    self.epochs = np.array(dat['epoch'])
	    self.values = np.array(dat[self.name_to_monitor])
	    self.dat = dat
	    print("loaded logs")
	    print(self.epochs)
	    print(self.values)
	else:
	    self.epochs = []
	    print("no logs yet")

    def get_changed(self):
	with open(self.changed_path, 'r') as file:
	    text = file.read()
	print("changed values: {}".format(text))
	self.changed = text
	return text

    def read_raw_logs(self):
	self.success = False
	self.finished = False
	with open(self.raw_logs_path, 'r') as file:
	    lines = file.readlines()
	if len(lines) > 1:
	    if lines[-1].strip() == 'done.':
		self.finished = True
	    	if lines[-2].strip() == 'finished.':
			self.success = True
	print('finished: {}, success: {}'.format(self.finished, self.success))

    def get_maximum(self):
	if len(self.epochs) > 0:
	    idx = np.argmax(self.values)	
	    print("maximum of {} at epoch {}".format(self.values[idx],self.epochs[idx]))
	    return self.epochs[idx],self.values[idx]
	else:
	    return -1,-1
	

def get_experiments(path):
    experiments = []
    num_tot = 0
    num_finished = 0
    num_success = 0
    for name in os.listdir(path):
	if os.path.isdir(path + name):
	    print(path+name)
	    exp= HyperparamExperiment(path+name)
	    num_finished += 1 if exp.finished else 0
	    num_success += 1 if exp.success else 0
	    num_tot += 1
	    experiments.append(exp)
    print("Read {} experiments, {} finished ({} success)".format(num_tot,num_finished,num_success))
    return experiments

experiments = get_experiments(dir_path)
best_experiment = np.argmax([e.get_maximum()[1] for e in experiments])
print("Best experiment so far: {}".format(experiments[best_experiment]))
