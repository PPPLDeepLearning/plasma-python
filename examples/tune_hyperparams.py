from plasma.primitives.hyperparameters import CategoricalHyperparam,ContinuousHyperparam,LogContinuousHyperparam,IntegerHyperparam
from plasma.utils.batch_jobs import create_slurm_script,create_sbatch_header,start_slurm_job,generate_working_dirname,copy_files_to_environment
from pprint import pprint
import yaml
import sys,os,getpass

tunables = []
shallow = True
num_nodes = 2
num_trials = 50

t_warn = CategoricalHyperparam(['data','T_warning'],[0.256,1.024,10.024])
cut_ends = CategoricalHyperparam(['data','cut_shot_ends'],[False,True])
#for shallow
if shallow:
    num_nodes = 1
    shallow_model = CategoricalHyperparam(['model','shallow_model','type'],["svm","random_forest","xgboost","mlp"])
    n_estimators = CategoricalHyperparam(['model','shallow_model','n_estimators'],[5,20,50,100,300,1000])
    max_depth = CategoricalHyperparam(['model','shallow_model','max_depth'],[0,3,6,10,30,100])
    C = LogContinuousHyperparam(['model','shallow_model','C'],1e-3,1e3)
    kernel = CategoricalHyperparam(['model','shallow_model','kernel'],["rbf","sigmoid","linear","poly"])
    xg_learning_rate = ContinuousHyperparam(['model','shallow_model','learning_rate'],0,1)
    scale_pos_weight = CategoricalHyperparam(['model','shallow_model','scale_pos_weight'],[1,10.0,100.0])
    num_samples = CategoricalHyperparam(['model','shallow_model','num_samples'],[10000,100000,1000000,1e7])
    hidden_size = CategoricalHyperparam(['model','final_hidden_layer_size'],[5,10,20])
    hidden_num = CategoricalHyperparam(['model','num_hidden_layers'],[2,4])
    mlp_learning_rate = CategoricalHyperparam(['model','learning_rate_mlp'],[0.001,0.0001,0.00001])
    tunables = [shallow_model,n_estimators,max_depth,C,kernel,xg_learning_rate,scale_pos_weight,num_samples,hidden_num,hidden_size,mlp_learning_rate] #target
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


def get_executable_name_imposed_shallow(shallow):
    from plasma.conf import conf
    if shallow:
        executable_name = conf['paths']['shallow_executable']
        use_mpi = False
    else:
        executable_name = conf['paths']['executable']
        use_mpi = True
    return executable_name,use_mpi



working_directory = generate_working_dirname(run_directory)
os.makedirs(working_directory)

executable_name,_ = get_executable_name_imposed_shallow(shallow)
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
