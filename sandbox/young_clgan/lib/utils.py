import multiprocessing

from rllab.sampler import parallel_sampler


def initialize_parallel_sampler(n_processes=-1):
    if n_processes == -1:
        n_processes = min(64, multiprocessing.cpu_count())
    parallel_sampler.initialize(n_parallel=n_processes)
