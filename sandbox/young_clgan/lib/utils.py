import multiprocessing

from rllab.sampler import parallel_sampler


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def initialize_parallel_sampler(n_processes=-1):
    if n_processes == -1:
        n_processes = min(64, multiprocessing.cpu_count())
    parallel_sampler.initialize(n_parallel=n_processes)
