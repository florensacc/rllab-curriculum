import datetime
import multiprocessing
import os
from rllab.sampler import parallel_sampler


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def initialize_parallel_sampler(n_processes=-1):
    if n_processes == -1:
        n_processes = min(64, multiprocessing.cpu_count())
    parallel_sampler.initialize(n_parallel=n_processes)
    
    
def format_experiment_prefix(exp_name):
    if ' ' in exp_name or len(exp_name) == 0:
        raise ValueError('Illegal experiment name!')
    exp_name = exp_name.replace('_', '-')
    return datetime.datetime.today().strftime('{}-%Y-%m-%d--%H-%M-%S'.format(exp_name))
    
    
def set_env_no_gpu():
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
