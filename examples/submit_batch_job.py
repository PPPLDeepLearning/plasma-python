from plasma.utils.batch_jobs import get_executable_name,generate_working_dirname,start_slurm_job,copy_files_to_environment
from pprint import pprint
import yaml
import sys,os,getpass

# tunables = []
# shallow = False
num_nodes = 2
num_trials = 1


run_directory = "/tigress/{}/batch_jobs/".format(getpass.getuser())
template_path = os.environ['PWD'] #"/home/{}/plasma-python/examples/".format(getpass.getuser())
conf_name = "conf.yaml"

def copy_conf_file(shallow,template_path = "../",save_path = "./",conf_name="conf.yaml"):
    assert(template_path != save_path)

    pathsrc = os.path.join(template_path,conf_name)
    pathdst = os.path.join(save_path,conf_name)

    with open(pathsrc, 'r') as yaml_file:
        conf = yaml.load(yaml_file)
    conf['training']['hyperparam_tuning'] = True #make sure all files like checkpoints and normalization are done locally
    with open(pathdst, 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)

def get_conf(template_path,conf_name):
    with open(os.path.join(template_path,conf_name), 'r') as yaml_file:
        conf = yaml.load(yaml_file)
    return conf

conf = get_conf(template_path,conf_name)
shallow = conf['model']['shallow']
if shallow:
    num_nodes = 1
working_directory = generate_working_dirname(run_directory)
os.makedirs(working_directory)

#copy conf and executable into directory
executable_name,_ = get_executable_name(conf)
os.system(" ".join(["cp -p",os.path.join(template_path,conf_name),working_directory]))
os.system(" ".join(["cp -p",os.path.join(template_path,executable_name),working_directory]))

os.chdir(working_directory)
print("Going into {}".format(working_directory))

for i in range(num_trials):
    subdir = working_directory + "/{}/".format(i) 
    os.makedirs(subdir)
    copy_files_to_environment(subdir)
    print("Making modified conf")
    copy_conf_file(shallow,working_directory,subdir,conf_name)
    print("Starting job")
    start_slurm_job(subdir,num_nodes,i,conf,shallow,conf['env']['name'])

print("submitted {} jobs.".format(num_trials))
