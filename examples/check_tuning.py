from plasma.primitives.hyperparameters import (
    # CategoricalHyperparam, ContinuousHyperparam, LogContinuousHyperparam,
    HyperparamExperiment
)
import sys
import os
import getpass
import numpy as np
import plasma.conf

dir_path = "/{}/{}/hyperparams/".format(
    plasma.conf.conf['fs_path'], getpass.getuser())
if len(sys.argv) <= 1:
    dir_path = dir_path + os.listdir(dir_path)[0] + '/'
    print("using default dir {}".format(dir_path))
else:
    dir_path = sys.argv[1]


def get_experiments(path):
    experiments = []
    num_tot = 0
    num_finished = 0
    num_success = 0
    for name in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, name)):
            print(os.path.join(path, name))
            exp = HyperparamExperiment(os.path.join(path, name))
            num_finished += 1 if exp.finished else 0
            num_success += 1 if exp.success else 0
            num_tot += 1
            experiments.append(exp)
    print("Read {} experiments, {} finished ({} success)".format(
        num_tot, num_finished, num_success))
    return experiments


experiments = sorted(get_experiments(dir_path))
print(len(experiments))
best_experiments = np.argsort(
    np.array([e.get_maximum(False)[0] for e in experiments]))
for e in experiments:
    e.summary()
print("Best experiment so far: \n")
for e in np.array(experiments)[best_experiments][-3:]:
    print(e)
