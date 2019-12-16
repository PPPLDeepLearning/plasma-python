from plasma.utils.batch_jobs import (
    generate_working_dirname, copy_files_to_environment, start_jenkins_job
    )
import yaml
import os
import getpass
import plasma.conf

num_nodes = 4  # Set in the Jenkins project area!!
test_matrix = [("Python3", "jet_data"), ("Python2", "jet_data")]

run_directory = "{}/{}/jenkins/".format(
    plasma.conf.conf['fs_path'], getpass.getuser())
template_path = os.environ['PWD']
conf_name = "conf.yaml"
executable_name = "mpi_learn.py"


def generate_conf_file(
        test_configuration,
        template_path="../",
        save_path="./",
        conf_name="conf.yaml"):
    assert(template_path != save_path)
    with open(os.path.join(template_path, conf_name), 'r') as yaml_file:
        conf = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    conf['training']['num_epochs'] = 2
    conf['paths']['data'] = test_configuration[1]
    if test_configuration[1] == "Python3":
        conf['env']['name'] = "PPPL_dev3"
        conf['env']['type'] = "anaconda3"
    else:
        conf['env']['name'] = "PPPL"
        conf['env']['type'] = "anaconda"

    with open(os.path.join(save_path, conf_name), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)
    return conf


working_directory = generate_working_dirname(run_directory)
os.makedirs(working_directory)

os.system(
    " ".join(["cp -p", os.path.join(template_path, conf_name),
              working_directory]))
os.system(" ".join(
    ["cp -p", os.path.join(template_path, executable_name),
     working_directory]))

# os.chdir(working_directory)
# print("Going into {}".format(working_directory))

for ci in test_matrix:
    subdir = working_directory + "/{}/".format(ci[0])
    os.makedirs(subdir)
    copy_files_to_environment(subdir)
    print("Making modified conf")
    conf = generate_conf_file(ci, working_directory, subdir, conf_name)
    print("Starting job")
    if ci[1] == "Python3":
        env_name = "PPPL_dev3"
        env_type = "anaconda3"
    else:
        env_name = "PPPL"
        env_type = "anaconda"
    start_jenkins_job(
        subdir,
        num_nodes,
        executable_name,
        ci,
        env_name,
        env_type)

print("submitted jobs.")
