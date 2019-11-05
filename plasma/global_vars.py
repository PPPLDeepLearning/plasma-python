import sys

# global variable defaults for non-MPI runs
comm = None
task_index = 0
num_workers = 1
NUM_GPUS = 0
MY_GPU = 0
backend = ''


def init_MPI(conf):
    from mpi4py import MPI
    global comm, task_index, num_workers
    global NUM_GPUS, MY_GPU, backend

    comm = MPI.COMM_WORLD
    task_index = comm.Get_rank()
    num_workers = comm.Get_size()

    NUM_GPUS = conf['num_gpus']
    MY_GPU = task_index % NUM_GPUS
    backend = conf['model']['backend']


def pprint_unique(obj):
    from pprint import pprint
    if task_index == 0:
        pprint(obj)


def print_unique(print_output, end='\n', flush=False):
    """
    Only master MPI rank 0 calls print().

    Trivial wrapper function to print()
    """
    if task_index == 0:
        print(print_output, end=end, flush=flush)


def write_unique(write_str):
    """
    Only master MPI rank 0 writes to and flushes stdout.

    A specialized case of print_unique(). Unlike print(), sys.stdout.write():
    - Must pass a string; will not cast argument
    - end='\n' kwarg of print() is not available
    (often the argument here is prepended with \r=carriage return in order to
    simulate a terminal output that overwrites itself)
    """
    # TODO(KGF): \r carriage returns appear as ^M in Unix-encoded .out files
    # from non-interactive Slurm batch jobs. Convert these to true Unix
    # line feeds / newlines (^J, \n) when we can detect such a stdout
    if task_index == 0:
        sys.stdout.write(write_str)
        sys.stdout.flush()


def write_all(write_str):
    '''All MPI ranks write to stdout, appending [rank]'''
    sys.stdout.write('[{}] '.format(task_index) + write_str)
    sys.stdout.flush()
