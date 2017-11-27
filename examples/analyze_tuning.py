from plasma.primitives.hyperparameters import CategoricalHyperparam,ContinuousHyperparam,LogContinuousHyperparam,HyperparamExperiment
import matplotlib.pylab as plt

from pprint import pprint
import yaml
import datetime
import uuid
import sys,os,getpass
import shutil
import subprocess as sp
import pandas
import numpy as np
import plasma.conf

dir_path = "/{}/{}/hyperparams/".format(plasma.conf.conf['fs_path'],getpass.getuser())

if len(sys.argv) <= 1:
    dir_path = dir_path + os.listdir(dir_path)[0] + '/'
    print("using default dir {}".format(dir_path))
else:
    dir_path = sys.argv[1]


def get_experiments(path,verbose=0):
    experiments = []
    num_tot = 0
    num_finished = 0
    num_success = 0
    for name in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path,name)):
            print(os.path.join(path,name))
            exp= HyperparamExperiment(os.path.join(path,name))
            num_finished += 1 if exp.finished else 0
            num_success += 1 if exp.success else 0
            num_tot += 1
            experiments.append(exp)
    if verbose: 
        print("Read {} experiments, {} finished ({} success)".format(num_tot,num_finished,num_success))
    return experiments

experiments = sorted(get_experiments(dir_path))
best_experiments = np.argsort(np.array([e.get_maximum(False)[0] for e in experiments]))
best = [] 
for e in np.array(experiments)[best_experiments][-5:]:
    best.append(e.get_number())

bigdict = {}
for base in best:
    f = "/{}/{}/changed_params.out".format(dir_path,base)
    data = open(f).readlines()

    for line in data:
        tuples = line.split(":")
        #if len(tuples) == 2:
        key, values = tuples[-2:]
        key = key.strip() 
        try:
            bigdict[key] += [values]
        except KeyError:
            bigdict[key] = [values]


def make_comparison_plot(key,tunable,trial):
    values,edges = tunable

    trial = list(map(lambda x: eval(x),trial))
    trial_values,_ = np.histogram(trial,bins=edges)
    total = trial_values.sum()
    values_percentages =list(map(lambda x: x*100.0/total, trial_values))

    plt.bar(edges[:-1], values_percentages, width=np.diff(edges), ec="k", align="edge")
    plt.xlabel(key, fontsize=20)
    #plt.yscale('log')
    plt.ylabel('Fraction of trials [%]/bin', fontsize=20)

    plt.bar(edges[:-1], values, width=np.diff(edges), ec="k", align="edge")
    plt.savefig(key+".png")
    #plt.show()
    plt.clf()


#default tunables:
defaults = {}
defaults['lr'] = np.histogram([1e-7,1e-4])
defaults['lr_decay'] = np.histogram([0.97,0.985,1.0])
defaults['positive_example_penalty'] = np.histogram([1.0,4.0,16.0])
#defaults['target'] = np.histogram([50,50,50],bins=['hinge','ttdinv','ttd'])
defaults['batch_size'] = np.histogram([64,256,1024])
defaults['dropout_prob'] = np.histogram([0.1,0.3,0.5])
defaults['rnn_layers'] = np.histogram([1,4])
defaults['rnn_size'] = np.histogram([100,200,300])
defaults['num_conv_filters'] = np.histogram([5,10])
defaults['num_conv_layers'] = np.histogram([2,4])
defaults['T_warning'] = np.histogram([0.256,1.024,10.024])
defaults['cut_shot_ends'] = np.histogram([False,True])

#Histogram it
for key,trial in bigdict.items():
    if key == 'target': continue
    make_comparison_plot(key,defaults[key],trial)
