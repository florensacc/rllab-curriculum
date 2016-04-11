from rllab.sampler.utils import rollout
from rllab.sampler.stateful_pool import singleton_pool
from rllab.misc import ext
from rllab.misc import logger
import numpy as np


def _worker_init(G, id):
    import os
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    G.worker_id = id


def initialize(n_parallel):
    singleton_pool.initialize(n_parallel)
    singleton_pool.run_each(
        _worker_init, [(id,) for id in xrange(singleton_pool.n_parallel)])


def _worker_populate_task(G, env, policy):
    G.env = env
    G.policy = policy


def populate_task(env, policy):
    logger.log("Populating workers...")
    singleton_pool.run_each(
        _worker_populate_task,
        [(env, policy)] * singleton_pool.n_parallel
    )
    logger.log("Populated")


def _worker_set_seed(_, seed):
    ext.set_seed(seed)


def set_seed(seed):
    singleton_pool.run_each(
        _worker_set_seed,
        [(seed + i,) for i in xrange(singleton_pool.n_parallel)]
    )


def _worker_set_policy_params(G, params):
    G.policy.set_param_values(params)


def _worker_collect_one_path(G, max_path_length):
    path = rollout(G.env, G.policy, max_path_length)
    return path, len(path["rewards"])


def sample_paths(
        policy_params,
        max_samples,
        max_path_length=np.inf):
    singleton_pool.run_each(
        _worker_set_policy_params,
        [(policy_params,)] * singleton_pool.n_parallel
    )
    return singleton_pool.run_collect(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(max_path_length,),
        show_prog_bar=True
    )
