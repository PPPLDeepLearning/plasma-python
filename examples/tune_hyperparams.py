from plasma.primitives.hyperparameters import CategoricalHyperparam,ContinuousHyperparam,LogContinuousHyperparam
from pprint import pprint
import yaml
import datetime
import uuid
import sys,os,getpass
import shutil
import subprocess as sp

tunables = []

lr = LogContinuousHyperparam(['model','lr'],5e-6,4e-3) 
lr_decay = CategoricalHyperparam(['model','lr_decay'],[0.97,0.985,1.0]) 
#t_warn = CategoricalHyperparam(['data','T_warning'],[1.024])
fac = CategoricalHyperparam(['data','positive_example_penalty'],[1.0,2.0,4.0,8.0])
#target = CategoricalHyperparam(['target'],['maxhinge','hinge'])
#batch_size = CategoricalHyperparam(['training','batch_size'],[256,128,32,64])
#dropout_prob = CategoricalHyperparam(['model','dropout_prob'],[0.1,0.3,0.5])


tunables = [lr,lr_decay,fac] #target

run_directory = "/tigress/{}/hyperparams/".format(getpass.getuser())
template_path = os.environ['PWD'] #"/home/{}/plasma-python/examples/".format(getpass.getuser())
conf_name = "conf.yaml"
executable_name = "mpi_learn.py"
num_machines = 2
num_trials = 50

def generate_conf_file(tunables,template_path = "../",save_path = "./",conf_name="conf.yaml"):
    assert(template_path != save_path)
    with open(os.path.join(template_path,conf_name), 'r') as yaml_file:
        conf = yaml.load(yaml_file)
    for tunable in tunables:
        tunable.assign_to_conf(conf,save_path)
    conf['training']['num_epochs'] = 1000 #rely on early stopping to terminate training
    conf['training']['hyperparam_tuning'] = True #rely on early stopping to terminate training
    with open(save_path+conf_name, 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)


def generate_working_dirname(run_directory):
    s = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    s += "_{}/".format(uuid.uuid4())
    return run_directory + s


def start_slurm_job(subdir,executable,num_machines,i):
    shutil.copy2(executable,subdir)
    script = create_slurm_script(subdir,num_machines,i)
    sp.Popen("sbatch "+script,shell=True)

def create_slurm_script(subdir,num_machines,idx):
    filename = "run_{}_nodes.cmd".format(num_machines)
    filepath = subdir+filename
    user = getpass.getuser()
    with open(filepath,"w") as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH -t 01:00:00\n')
        f.write('#SBATCH -N '+str(num_machines)+'\n')
        f.write('#SBATCH --ntasks-per-node=4\n')
        f.write('#SBATCH --ntasks-per-socket=2\n')
        f.write('#SBATCH --gres=gpu:4\n')
        f.write('#SBATCH -c 4\n')
        f.write('#SBATCH --mem-per-cpu=0\n')
        f.write('#SBATCH -o {}.out\n'.format(idx))
        f.write('\n\n')
        f.write('module load anaconda3\n')
        f.write('source activate PPPL_dev3\n')
        f.write('module load cudatoolkit/8.0 cudnn/cuda-8.0/6.0 openmpi/cuda-8.0/intel-17.0/2.1.0/64 intel/17.0/64/17.0.4.196 intel-mkl/2017.3/4/64\n')
        f.write('rm -f /tigress/{}/model_checkpoints/*.h5\n'.format(user))
        f.write('cd {}\n'.format(subdir))
        f.write('export OMPI_MCA_btl=\"tcp,self,sm\"\n')
        f.write('srun python mpi_learn.py\n')    
        f.write('echo "done."')

    return filepath 

def copy_files_to_environment(subdir):
    from plasma.conf import conf
    normalization_dir = os.path.dirname(conf['paths']['normalizer_path'])
    if os.path.isdir(normalization_dir):
        print("Copying normalization to")
        shutil.copytree(normalization_dir,subdir+os.path.basename(normalization_dir))


working_directory = generate_working_dirname(run_directory)
os.makedirs(working_directory)
shutil.copy2(os.path.join(template_path,conf_name),working_directory)
shutil.copy2(os.path.join(template_path+executable_name,working_directory)
os.chdir(working_directory)
print("Going into {}".format(working_directory))

for i in range(num_trials):
    print(i)
    subdir = working_directory + "/{}/".format(i) 
    os.makedirs(subdir)
    copy_files_to_environment(subdir)
    print("Making modified conf")
    generate_conf_file(tunables,working_directory,subdir,conf_name)
    print("Starting job")
    start_slurm_job(subdir,executable_name,num_machines,i)

print("submitted {} jobs.".format(num_trials))
