import pandas as pd
import glob
from subprocess import Popen
import yaml
import os
import math
import numpy as np
from random import shuffle
from joblib import Parallel, delayed
import multiprocessing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import pdb

def arrangeTrialsAtRandom(filenames,scale=1.0):
    shuffle(filenames)
    previous = pd.read_csv(filenames[0])
    previous['times'] = previous['times'].apply(lambda x: x/60.0/scale)
    dataframes = [previous]
    for filename in filenames[1:]:
        shift = max(previous['times'].values)
        current = pd.read_csv(filename)
        current['times'] = current['times'].apply(lambda x: x/60.0/scale+shift)
        dataframes.append(current)
        previous = current
    return pd.concat(dataframes)

def getOneBestValidationAUC(T_of_test,dataset):
    #select subset of dataframe by time for all
    dataset = dataset[dataset.times <= T_of_test]
 
    #apply emulate_converge script
    aucs = dataset['val_roc'].values
    if len(aucs) > 0:
        return max(aucs)
    else:
        return 0.0

def doPlot(parallel_aucs, serial_aucs, times, errors):
    times = list(times)
    times_histo = np.histogram(parallel_aucs,bins=times)
    #values,edges = times_histo
    parallel_values = parallel_aucs[1:]
    edges = times
    print(len(parallel_values))
    print(len(edges))
    serial_values = np.array(serial_aucs[1:])
    errors = np.array(errors[1:])
    edges = np.array(times[:-1])
    print(errors.shape)
    print(edges.shape)
    print(serial_values.shape)


    plt.figure()
    plt.plot(edges, parallel_values,label = "Distributed search") #, width=np.diff(edges), ec="k", align="edge")
    plt.plot(edges, serial_values, label="Sequential search") #, width=np.diff(edges), ec="k", align="edge")
    #plt.fill_between(edges, serial_values-errors,serial_values+errors)
    plt.legend(loc = (0.6,0.7))
    plt.xlabel("Time [minutes]", fontsize=20)
    #plt.yscale('log')
    plt.ylabel('Best validation AUC', fontsize=20)
    plt.savefig("times.png")

    plt.figure()
    plt.plot(edges, parallel_values,label = "Distributed search") #, width=np.diff(edges), ec="k", align="edge")
    plt.plot(edges, serial_values, label="Sequential search") #, width=np.diff(edges), ec="k", align="edge")
    #plt.fill_between(edges, serial_values-errors,serial_values+errors)
    plt.legend(loc = (0.6,0.7))
    plt.xlabel("Time [minutes]", fontsize=20)
    plt.xscale('log')
    plt.xlim([0,100])
    plt.ylabel('Best validation AUC', fontsize=20)
    plt.savefig("times_logx_start.png")

    plt.figure()
    plt.plot(edges, parallel_values,label = "Distributed search") #, width=np.diff(edges), ec="k", align="edge")
    plt.plot(edges, serial_values, label="Sequential search") #, width=np.diff(edges), ec="k", align="edge")
    #plt.fill_between(edges, serial_values-errors,serial_values+errors)
    plt.legend(loc = (0.6,0.7))
    plt.xlabel("Time [minutes]", fontsize=20)
    plt.xscale('log')
    plt.xlim([100,10000])
    plt.ylabel('Best validation AUC', fontsize=20)
    plt.savefig("times_logx.png")


def getReplica(filenames, times):
    serial_auc_replica = arrangeTrialsAtRandom(filenames,100.0)

    best_serial_aucs_over_time = []
    for T in times:
        current_best = 0
        ##pass AUCs and real epoch counts to emulate_converge
        auc = getOneBestValidationAUC(T,serial_auc_replica)
        if auc > current_best: current_best = auc

        best_serial_aucs_over_time.append(current_best)

    #replicas.append(best_serial_aucs_over_time)
    return best_serial_aucs_over_time

def getTimeReplica(filenames,T):
    current_best = 0
    for filename in filenames:
        #get AUCs for this trial, one per effective epoch
        try:
            dataset = pd.read_csv(filename)
            dataset['times'] = dataset['times'].apply(lambda x: x/60.0)
        except:
            print("No data in {}".format(filename))
            continue
        ##pass AUCs and real epoch counts to emulate_converge
        auc = getOneBestValidationAUC(T,dataset)
        if auc > current_best: current_best = auc
    return current_best

def getTimeReplicaSerial(serial_auc_replica,T):
    current_best = 0
    ##pass AUCs and real epoch counts to emulate_converge
    auc = getOneBestValidationAUC(T,serial_auc_replica)
    if auc > current_best: current_best = auc

    #replicas.append(best_serial_aucs_over_time)
    return current_best


if __name__ == '__main__':

    filenames = glob.glob("/tigress/FRNN/JET_Titan_hyperparameter_run/*/temporal_csv_log.csv")
    patience = 5

    times = np.linspace(0,310*30,186*30)

    best_parallel_aucs_over_time = []
    num_cores = multiprocessing.cpu_count()
    print ("Running on ", num_cores, " CPU cores")
    best_parallel_aucs_over_time = Parallel(n_jobs=num_cores)(delayed(getTimeReplica)(filenames, T) for T in times) 

    Nreplicas = 20
    replicas = []


    for i in range(Nreplicas):
        serial_auc_replica = arrangeTrialsAtRandom(filenames,100.0)

        #replicas = Parallel(n_jobs=num_cores)(delayed(getReplica)(filenames, times) for i in range(Nreplicas)) 
        best_serial_aucs_over_time = Parallel(n_jobs=num_cores)(delayed(getTimeReplicaSerial)(serial_auc_replica, T) for T in times)
        replicas.append(best_serial_aucs_over_time)


    from statistics import mean,stdev
    best_serial_aucs_over_time = list(map(mean, zip(*replicas)))
    errors = list(map(stdev, zip(*replicas)))

    doPlot(best_parallel_aucs_over_time, best_serial_aucs_over_time, times, errors)
