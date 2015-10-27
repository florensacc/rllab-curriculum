#!/usr/bin/python
import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
import multiprocessing
from sampler import parallel_sampler
parallel_sampler.init_pool(multiprocessing.cpu_count())
#parallel_sampler.init_pool(1)

from qfunc import AtariRAMQFunction
from algo.pfqi import PFQI
from mdp import AtariMDP
import numpy as np

np.random.seed(0)

if __name__ == '__main__':

    mdp = AtariMDP(rom_path="vendor/atari_roms/pong.bin", obs_type='ram')
    qfunc = AtariRAMQFunction(mdp, hidden_sizes=[256, 128])
    algo = PFQI(
        samples_per_itr=50000,
        test_samples_per_itr=50000,
        stepsize=0.5,
        penalty_expand_factor=1.3,
        penalty_shrink_factor=0.75,
        max_epsilon=0.1,
        min_epsilon=0.1,
    )
    algo.train(mdp, qfunc)
