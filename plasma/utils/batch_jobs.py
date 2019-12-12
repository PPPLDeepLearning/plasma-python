from __future__ import division
import datetime
import uuid
import os
# import getpass
import subprocess as sp


def generate_working_dirname(run_directory):
    s = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    s += "_{}".format(uuid.uuid4())
    return run_directory + s


def get_executable_name(conf):
    shallow = conf['model']['shallow']
    if shallow:
        executable_name = conf['paths']['shallow_executable']
        use_mpi = False
    else:
        executable_name = conf['paths']['executable']
        use_mpi = True
    return executable_name, use_mpi


def start_slurm_job(subdir, num_nodes, i, conf, shallow,
                    env_name="frnn", env_type="anaconda"):
    executable_name, use_mpi = get_executable_name(conf)
    os.system(" ".join(["cp -p", executable_name, subdir]))
    script = create_slurm_script(subdir, num_nodes, i, executable_name,
                                 use_mpi, env_name, env_type)
    sp.Popen("sbatch " + script, shell=True)


def create_jenkins_script(subdir, num_nodes, executable_name,
                          test_configuration, env_name="frnn",
                          env_type="anaconda"):
    filename = "jenkins_{}_{}.cmd".format(
        test_configuration[0],
        test_configuration[1])  # version of Python and the dataset
    filepath = os.path.join(subdir, filename)
    # user = getpass.getuser()
    with open(filepath, "w") as f:
        f.write('#!/usr/bin/bash\n')
        f.write('export OMPI_MCA_btl=\"tcp,self,sm\"\n')
        f.write('echo \"Jenkins test {}\"\n'.format(test_configuration[1]))
        f.write('rm /tigress/alexeys/model_checkpoints/*\n')
        f.write('rm -rf /tigress/alexeys/processed_shots\n')
        f.write('rm -rf /tigress/alexeys/processed_shotlists\n')
        f.write('rm -rf /tigress/alexeys/normalization\n')
        f.write('module load {}\n'.format(env_type))
        f.write('source activate {}\n'.format(env_name))
        f.write('module load cudatoolkit/8.0\n')
        f.write('module load cudnn/cuda-8.0/6.0\n')
        f.write('module load openmpi/cuda-8.0/intel-17.0/2.1.0/64\n')
        f.write('module load intel/17.0/64/17.0.4.196\n')
        f.write('cd /home/alexeys/jenkins/workspace/FRNM/PPPL\n')
        f.write('python setup.py install\n')
        f.write('cd {}\n'.format(subdir))
        f.write('srun -N {} -n {} python {}\n'.format(
            num_nodes // 2, num_nodes // 2 * 4, executable_name))
    return filepath


def start_jenkins_job(subdir, num_nodes, executable_name, test_configuration,
                      env_name, env_type):
    os.system(" ".join(["cp -p", executable_name, subdir]))
    script = create_jenkins_script(subdir, num_nodes, executable_name,
                                   test_configuration, env_name, env_type)
    sp.Popen("sh " + script, shell=True)


def start_pbs_job(subdir, num_nodes, i, conf, shallow,
                  env_name="frnn", env_type="anaconda"):
    executable_name, use_mpi = get_executable_name(conf)
    os.system(" ".join(["cp -p", executable_name, subdir]))
    script = create_pbs_script(subdir, num_nodes, i, executable_name, use_mpi,
                               env_name, env_type)
    sp.Popen("qsub " + script, shell=True)


def create_slurm_script(subdir, num_nodes, idx, executable_name, use_mpi,
                        env_name="frnn", env_type="anaconda3"):
    filename = "run_{}_nodes.cmd".format(num_nodes)
    filepath = subdir + filename
    # user = getpass.getuser()
    sbatch_header = create_slurm_header(num_nodes, use_mpi, idx)
    with open(filepath, "w") as f:
        for line in sbatch_header:
            f.write(line)
        f.write('module load ' + env_type + '\n')
        f.write('conda activate ' + env_name + '\n')
        f.write((
            'module load cudatoolkit \n'))
        f.write((
            'module load openmpi/gcc/3.1.4/64 \n'))
        f.write((
            'module load hdf5/gcc/openmpi-3.1.4/1.10.5 \n'))
        # f.write('rm -f /tigress/{}/model_checkpoints/*.h5\n'.format(user))
        f.write('srun python {}\n'.format(executable_name))
        f.write('echo "done."')

    return filepath


def create_pbs_script(subdir, num_nodes, idx, executable_name, use_mpi,
                      env_name="frnn", env_type="anaconda"):
    filename = "run_{}_nodes.cmd".format(num_nodes)
    filepath = subdir + filename
    # user = getpass.getuser()
    sbatch_header = create_pbs_header(num_nodes, use_mpi, idx)
    with open(filepath, "w") as f:
        for line in sbatch_header:
            f.write(line)
        # f.write('export HOME=/lustre/atlas/proj-shared/fus117\n')
        # f.write('cd $HOME/PPPL/plasma-python/examples\n')
        f.write('source $MODULESHOME/init/bash\n')
        f.write('module load tensorflow\n')
        # f.write('rm $HOME/tigress/alexeys/model_checkpoints/*\n')
        f.write('cd {}\n'.format(subdir))
        f.write('aprun -n {} -N1 env KERAS_HOME={} singularity exec '
                '$TENSORFLOW_CONTAINER python3 {}\n'.format(
                    str(num_nodes), subdir, executable_name))
        f.write('echo "done."')

    return filepath


def create_slurm_header(num_nodes, use_mpi, idx):
    if not use_mpi:
        assert num_nodes == 1
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('#SBATCH -t 24:00:00\n')
    lines.append('#SBATCH -N ' + str(num_nodes) + '\n')
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


def create_pbs_header(num_nodes, use_mpi, idx):
    if not use_mpi:
        assert num_nodes == 1
    lines = []
    lines.append('#!/bin/bash\n')

    lines.append('#PBS -A FUS117\n')
    lines.append('#PBS -l walltime=02:00:00\n')
    lines.append('#PBS -l nodes=' + str(num_nodes) + '\n')
    lines.append('#PBS -o {}.out\n'.format(idx))
    lines.append('\n\n')
    return lines


def copy_files_to_environment(subdir):
    from plasma.conf import conf
    normalization_dir = os.path.dirname(
        conf['paths']['global_normalizer_path'])
    if os.path.isdir(normalization_dir):
        print("Copying normalization to")
        os.system(" ".join(
            ["cp -rp", normalization_dir,
             os.path.join(subdir, os.path.basename(normalization_dir))]))
