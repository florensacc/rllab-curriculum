#!/usr/bin/python
import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
import multiprocessing
from sampler import parallel_sampler
parallel_sampler.init_pool(multiprocessing.cpu_count())
#parallel_sampler.init_pool(1)

from misc.overrides import overrides
from qfunc import LasagneQFunction
from algo.bpfqi import BPFQI
from mdp import FrozenLakeMDP
import numpy as np
from core.serializable import Serializable
import cgtcompat as theano
import cgtcompat.tensor as T
import lasagne.layers as L
from qfunc import TabularQFunction

np.random.seed(0)

if __name__ == '__main__':

    desc = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]

    map8x8 = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]

    mdp = FrozenLakeMDP(map8x8)
    qfunc = TabularQFunction(mdp)
    algo = BPFQI(
        samples_per_itr=20000,
        max_path_length=100,
        test_samples_per_itr=20000,
        stepsize=0.01,
        penalty_expand_factor=2,
        penalty_shrink_factor=0.5,
        adapt_penalty=True,
        initial_penalty=1,
        max_penalty_itr=3,
    )
    algo.train(mdp, qfunc)
