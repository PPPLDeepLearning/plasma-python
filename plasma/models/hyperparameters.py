import numpy as np
import random
import abc

class Hyperparam(object):

    @abc.abstractmethod
    def choice(self):
    	return 0

    def get_conf_entry(self,conf):
    	el = conf
    	for sub_path in self.path:
    		el = el[sub_path]
    	return el

    def assign_to_conf(self,conf,save_path):
    	val = self.choice()
    	print(" : ".join(self.path)+ ": {}".format(val))
    	el = conf
    	for sub_path in self.path[:-1]:
    		el = el[sub_path]
    	el[self.path[-1]] = val

    	with open(save_path+"changed_params.out", 'a+') as outfile:
    		for el in self.path:
	    		outfile.write("{} : ".format(el))
	    	outfile.write("{}\n".format(val))



class CategoricalHyperparam(Hyperparam):

	def __init__(self,path,values):
		self.path = path
		self.values = values

	def choice(self):
		return random.choice(self.values)


class ContinuousHyperparam(Hyperparam):
	def __init__(self,path,lo,hi):
		self.path = path
		self.lo =lo 
		self.hi =hi 

	def choice(self):
		return float(np.random.uniform(self.lo,self.hi))

class LogContinuousHyperparam(Hyperparam):
	def __init__(self,path,lo,hi):
		self.path = path
		self.lo = self.to_log(lo)
		self.hi = self.to_log(hi) 

	def to_log(self,num_val):
		return np.log10(num_val)

	def choice(self):
		return float(np.power(10,np.random.uniform(self.lo,self.hi)))


class IntegerHyperparam(Hyperparam):
	def __init__(self,path,lo,hi):
		self.path = path
		self.lo =lo 
		self.hi =hi 

	def choice(self):
		return int(np.random.random_integers(self.lo,self.hi))


class GenericHyperparam(Hyperparam):
	def __init__(self,path,choice_fn):
		self.path = path
		self.choice_fn = choice_fn

	def choice(self):
		return self.choice_fn()
