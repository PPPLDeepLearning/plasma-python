from __future__ import print_function
from mpi4py import MPI

from hostlist import expand_hostlist
import socket
import os
import math


def get_host_to_id_mapping():
    return {host: i for (i, host) in enumerate(
        expand_hostlist(os.environ['SLURM_NODELIST']))}


def get_my_host_id():
    return get_host_to_id_mapping()[socket.gethostname()]


def get_host_list(port):
    return ['{}:{}'.format(host, port) for
            host in expand_hostlist(os.environ['SLURM_NODELIST'])]


def get_worker_host_list(base_port, workers_per_host):
    hosts = expand_hostlist(os.environ['SLURM_NODELIST'])
    ports = [base_port + i for i in range(workers_per_host)]
    worker_hlist = []
    for h in hosts:
        for p in ports:
            worker_hlist.append('{}:{}'.format(h, p))
    return worker_hlist


def get_worker_host(base_port, workers_per_host, task_id):
    return get_worker_host_list(base_port, workers_per_host)[task_id]


def get_ps_host_list(base_port, num_ps):
    assert num_ps < 10000000
    port = base_port
    ps_hlist = []
    hosts = expand_hostlist(os.environ['SLURM_NODELIST'])
    while True:
        for host in hosts:
            if len(ps_hlist) >= num_ps:
                return ps_hlist
            ps_hlist.append('{}:{}'.format(host, port))
        port += 1


def get_ps_host(base_port, num_ps, num_workers, task_id):
    return get_ps_host_list(base_port, num_ps)[task_id - num_workers]


def get_mpi_cluster_server_jobname(num_ps=1, num_workers=None):
    import tensorflow as tf
    NUM_GPUS = 4
    comm = MPI.COMM_WORLD
    task_index = comm.Get_rank()
    task_num = comm.Get_size()
    num_hosts = len(get_host_list(1))

    num_workers_per_host = NUM_GPUS
    # TODO(KGF): this error handling is completely duplicated below
    if (num_workers is None or num_workers == 0
            or num_workers > num_workers_per_host*num_hosts):
        if num_workers > num_workers_per_host*num_hosts:
            print('Num workers too large (more than one per GPU). ',
                  'Setting to default of {} for {} hosts'.format(
                      num_workers_per_host * num_hosts, num_hosts))
        if num_workers == 0:
            print('Num workers set to 0, should be positive. ',
                  'Setting to default of {} for {} hosts'.format(
                      num_workers_per_host * num_hosts, num_hosts))
        num_workers = num_workers_per_host*num_hosts

    tasks_per_node = task_num / num_hosts

    max_ps = num_hosts*(tasks_per_node - num_workers_per_host)
    print("tasks_per_node {} num_workers_per_host {} num_hosts {}".format(
        tasks_per_node, num_workers_per_host, num_hosts))
    if num_ps == 0 or num_ps > max_ps:
        print('Invalid number of ps {} (maximum {}, minimum 0)'.format(
            num_ps, max_ps))
        if num_ps == 0:
            print('Setting to 1')
            num_ps = 1
        else:
            print('Setting to {}'.format(max_ps))
            num_ps = max_ps
    num_ps_per_host = int(math.ceil(1.0*num_ps/num_hosts))
    task_index = task_index % tasks_per_node

    if task_index < NUM_GPUS:
        job_name = 'worker'
        num_per_host = num_workers_per_host
        global_task_index = get_my_host_id()*num_per_host+task_index
    else:
        job_name = 'ps'
        task_index = task_index - NUM_GPUS
        num_per_host = num_ps_per_host
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        global_task_index = num_hosts*task_index+get_my_host_id()

    #     if job_name == "ps":
    # if job_name == "worker":
    #   os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(MY_GPU)
    num_total = num_workers + num_ps
    assert task_num >= num_total

    if task_index == 0:
        print('{} superfluous processes'.format(task_num - num_total))
        print('{} superfluous workers'.format(
            num_workers_per_host * num_hosts - num_workers))
        print('{} superfluous ps'.format(num_ps_per_host*num_hosts - num_ps))

    print('{}, task_id: {}, host_id: {}'.format(
        socket.gethostname(), task_index, get_my_host_id()))

    worker_hosts = get_worker_host_list(
        2222, num_workers_per_host)[:num_workers]
    ps_hosts = get_ps_host_list(2322, num_ps)
    if global_task_index == 0:
        print('ps_hosts: {}\n, worker hosts: {}\n'.format(
            ps_hosts, worker_hosts))
    # Create a cluster from the parameter server and worker hosts.
    if job_name == 'ps' and global_task_index >= num_ps:
        exit(0)
    if job_name == 'worker' and global_task_index >= num_workers:
        exit(0)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=job_name,
                             task_index=global_task_index)

    return cluster, server, job_name, global_task_index, num_workers


def get_mpi_task_index(num_workers=None):
    NUM_GPUS = 4
    comm = MPI.COMM_WORLD
    task_index = comm.Get_rank()
    task_num = comm.Get_size()
    num_hosts = len(get_host_list(1))

    num_workers_per_host = NUM_GPUS
    if (num_workers is None or num_workers == 0
            or num_workers > num_workers_per_host*num_hosts):
        if num_workers > num_workers_per_host*num_hosts:
            print('Num workers too large (more than one per GPU). ',
                  'Setting to default of {} for {} hosts'.format(
                      num_workers_per_host * num_hosts, num_hosts))
        if num_workers == 0:
            print('Num workers set to 0, should be positive. ',
                  'Setting to default of {} for {} hosts'.format(
                      num_workers_per_host * num_hosts, num_hosts))
        num_workers = num_workers_per_host*num_hosts

    tasks_per_node = task_num / num_hosts
    task_index = task_index % tasks_per_node

    if task_index < NUM_GPUS:
        job_name = 'worker'
        num_per_host = num_workers_per_host
        global_task_index = get_my_host_id()*num_per_host+task_index
    else:
        exit(0)

    num_total = num_workers
    assert task_num >= num_total

    if task_index == 0:
        print('{} superfluous workers'.format(
            num_workers_per_host * num_hosts - num_workers))
        print('{} total workers'.format(num_workers))

    print('{}, task_id: {}, host_id: {}'.format(
        socket.gethostname(), task_index, get_my_host_id()))
    if job_name == 'worker' and global_task_index >= num_workers:
        exit(0)

    return global_task_index, num_workers
