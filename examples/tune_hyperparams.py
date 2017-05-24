from plasma.models.hyperparameters import CategoricalHyperparam,ContinuousHyperparam,LogContinuousHyperparam
from pprint import pprint
import yaml

tunables = []

lr = LogContinuousHyperparam(['model','lr'],1e-7,1e-2) 
# lr = CategoricalHyperparam(['model','lr'],[0.001,0.01,0.1]) 
t_warn = CategoricalHyperparam(['data','T_warning'],[0.128,0.512])
target = CategoricalHyperparam(['target'],['lasso','hi'])


tunables = [lr,t_warn,target]

def generate_conf_file(tunables,template_path = "./conf_template.yaml",save_path = "./conf.yaml"):
	with open(template_path, 'r') as yaml_file:
		conf = yaml.load(yaml_file)
	for tunable in tunables:
		tunable.assign_to_conf(conf)
	with open(save_path, 'w') as outfile:
		yaml.dump(conf, outfile, default_flow_style=False)



generate_conf_file(tunables)


