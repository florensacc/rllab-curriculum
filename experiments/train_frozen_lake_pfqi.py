#!/usr/bin/python
import os
os.environ['TENSORFUSE_MODE'] = 'theano'
import multiprocessing
from sampler import parallel_sampler
parallel_sampler.init_pool(multiprocessing.cpu_count())
#parallel_sampler.init_pool(1)

from misc.overrides import overrides
from qfunc import LasagneQFunction
from algo.pfqi import PFQI
from mdp import FrozenLakeMDP
import numpy as np
from core.serializable import Serializable
import tensorfuse as theano
import tensorfuse.tensor as T
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
    mdp = FrozenLakeMDP(desc)
    qfunc = TabularQFunction(mdp)
    algo = PFQI(
        samples_per_itr=50000,
        max_path_length=100,
        test_samples_per_itr=50000,
        stepsize=0.5,
        penalty_expand_factor=1.3,
        penalty_shrink_factor=0.75,
        adapt_penalty=True,
        initial_penalty=1,
    )
    algo.train(mdp, qfunc)
