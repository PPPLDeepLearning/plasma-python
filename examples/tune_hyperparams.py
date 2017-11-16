from plasma.primitives.hyperparameters import CategoricalHyperparam,ContinuousHyperparam,LogContinuousHyperparam,IntegerHyperparam
from pprint import pprint
import yaml
import datetime
import uuid
import sys,os,getpass
import subprocess as sp
import numpy as np

tunables = []
shallow = True
num_nodes = 2
num_trials = 50

t_warn = CategoricalHyperparam(['data','T_warning'],[0.256,1.024,10.024])
cut_ends = CategoricalHyperparam(['data','cut_shot_ends'],[False,True])
#for shallow
if shallow:
    num_nodes = 1
    shallow_model = CategoricalHyperparam(['model','shallow_model','type'],["svm","random_forest","xgboost"])
    n_estimators = CategoricalHyperparam(['model','shallow_model','n_estimators'],[5,20,50,100,300,1000])
    max_depth = CategoricalHyperparam(['model','shallow_model','max_depth'],[0,3,6,10,30,100])
    C = LogContinuousHyperparam(['model','shallow_model','C'],1e-3,1e3)
    kernel = CategoricalHyperparam(['model','shallow_model','kernel'],["rbf","sigmoid","linear","poly"])
    xg_learning_rate = ContinuousHyperparam(['model','shallow_model','learning_rate'],0,1)
    scale_pos_weight = CategoricalHyperparam(['model','shallow_model','scale_pos_weight'],[1,10.0,100.0])
    num_samples = CategoricalHyperparam(['model','shallow_model','num_samples'],[10000,100000,1000000,1e7])
    tunables = [shallow_model,n_estimators,max_depth,C,kernel,xg_learning_rate,scale_pos_weight,num_samples] #target
else:
    #for DL
    lr = LogContinuousHyperparam(['model','lr'],1e-7,1e-4)
    lr_decay = CategoricalHyperparam(['model','lr_decay'],[0.97,0.985,1.0])
    fac = CategoricalHyperparam(['data','positive_example_penalty'],[1.0,4.0,16.0])
    target = CategoricalHyperparam(['target'],['maxhinge','hinge','ttdinv','ttd'])
    batch_size = CategoricalHyperparam(['training','batch_size'],[64,256,1024])
    dropout_prob = CategoricalHyperparam(['model','dropout_prob'],[0.1,0.3,0.5])
    conv_filters = CategoricalHyperparam(['model','num_conv_filters'],[5,10])
    conv_layers = IntegerHyperparam(['model','num_conv_layers'],2,4)
    rnn_layers = IntegerHyperparam(['model','rnn_layers'],1,4)
    rnn_size = CategoricalHyperparam(['model','rnn_size'],[100,200,300])
    tunables = [lr,lr_decay,fac,target,batch_size,dropout_prob]
    tunables += [conv_filters,conv_layers,rnn_layers,rnn_size]
tunables += [cut_ends,t_warn]


run_directory = "/tigress/{}/hyperparams/".format(getpass.getuser())
template_path = os.environ['PWD'] #"/home/{}/plasma-python/examples/".format(getpass.getuser())
conf_name = "conf.yaml"

def generate_conf_file(tunables,shallow,template_path = "../",save_path = "./",conf_name="conf.yaml"):
    assert(template_path != save_path)
    with open(os.path.join(template_path,conf_name), 'r') as yaml_file:
        conf = yaml.load(yaml_file)
    for tunable in tunables:
        tunable.assign_to_conf(conf,save_path)
    conf['training']['num_epochs'] = 1000 #rely on early stopping to terminate training
    conf['training']['hyperparam_tuning'] = True #rely on early stopping to terminate training
    conf['model']['shallow'] = shallow
    with open(os.path.join(save_path,conf_name), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)
    return conf


def generate_working_dirname(run_directory):
    s = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    s += "_{}".format(uuid.uuid4())
    return run_directory + s

def get_executable_name(shallow):
    from plasma.conf import conf
    if shallow:
        executable_name = conf['paths']['shallow_executable']
        use_mpi = False
    else:
        executable_name = conf['paths']['executable']
        use_mpi = True
    return executable_name,use_mpi


def start_slurm_job(subdir,num_nodes,i,conf,shallow):
    executable_name,use_mpi = get_executable_name(shallow)
    os.system(" ".join(["cp -p",executable_name,subdir]))
    script = create_slurm_script(subdir,num_nodes,i,executable_name,use_mpi)
    sp.Popen("sbatch "+script,shell=True)

def create_slurm_script(subdir,num_nodes,idx,executable_name,use_mpi):
    filename = "run_{}_nodes.cmd".format(num_nodes)
    filepath = subdir+filename
    user = getpass.getuser()
    sbatch_header = create_sbatch_header(num_nodes,use_mpi,idx)
    with open(filepath,"w") as f:
        for line in sbatch_header:
            f.write(line)
        f.write('module load anaconda\n')
        f.write('source activate frnn\n')
        f.write('module load cudatoolkit/8.0 cudnn/cuda-8.0/6.0 openmpi/cuda-8.0/intel-17.0/2.1.0/64 intel/17.0/64/17.0.4.196 intel-mkl/2017.3/4/64\n')
        # f.write('rm -f /tigress/{}/model_checkpoints/*.h5\n'.format(user))
        f.write('cd {}\n'.format(subdir))
        f.write('export OMPI_MCA_btl=\"tcp,self,sm\"\n')
        f.write('srun python {}\n'.format(executable_name))
        f.write('echo "done."')

    return filepath

def create_sbatch_header(num_nodes,use_mpi,idx):
    if not use_mpi:
        assert(num_nodes == 1)
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('#SBATCH -t 04:00:00\n')
    lines.append('#SBATCH -N '+str(num_nodes)+'\n')
    if use_mpi:
        lines.append('#SBATCH --ntasks-per-node=4\n')
        lines.append('#SBATCH --ntasks-per-socket=2\n')
    else:
        lines.append('#SBATCH --ntasks-per-node=1\n')
        lines.append('#SBATCH --ntasks-per-socket=1\n')
    lines.append('#SBATCH --gres=gpu:4\n')
    lines.append('#SBATCH -c 4\n')
    lines.append('#SBATCH --mem-per-cpu=0\n')
    lines.append('#SBATCH -o {}.out\n'.format(idx))
    lines.append('\n\n')
    return lines

def copy_files_to_environment(subdir):
    from plasma.conf import conf
    normalization_dir = os.path.dirname(conf['paths']['normalizer_path'])
    if os.path.isdir(normalization_dir):
        print("Copying normalization to")
        os.system(" ".join(["cp -rp",normalization_dir,os.path.join(subdir,os.path.basename(normalization_dir))]))

working_directory = generate_working_dirname(run_directory)
os.makedirs(working_directory)

executable_name,_ = get_executable_name(shallow)
os.system(" ".join(["cp -p",os.path.join(template_path,conf_name),working_directory]))
os.system(" ".join(["cp -p",os.path.join(template_path,executable_name),working_directory]))

os.chdir(working_directory)
print("Going into {}".format(working_directory))

for i in range(num_trials):
    subdir = working_directory + "/{}/".format(i) 
    os.makedirs(subdir)
    copy_files_to_environment(subdir)
    print("Making modified conf")
    conf = generate_conf_file(tunables,shallow,working_directory,subdir,conf_name)
    print("Starting job")
    start_slurm_job(subdir,num_nodes,i,conf,shallow)

print("submitted {} jobs.".format(num_trials))
