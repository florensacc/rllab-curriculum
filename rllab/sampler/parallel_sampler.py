from joblib.pool import MemmapingPool
from rllab.sampler.utils import rollout, ProgBarCounter
from rllab.misc.ext import extract
from multiprocessing import Manager
import numpy as np
import traceback
import sys

__all__ = [
    'init_pool',
    'populate_task',
    'reset',
]

_mdp = None
_policy = None
_n_parallel = 1
_pool = None


def pool_populate_task(args):
    global _mdp, _policy
    _mdp, _policy = args


def pool_rollout(args):
    try:
        policy_params, max_samples, max_path_length, queue, record_states, whole_paths = \
            extract(
                args,
                "policy_params", "max_samples", "max_path_length", "queue",
                "record_states", "whole_paths"
            )
        _policy.set_param_values(policy_params)
        n_samples = 0
        paths = []
        if queue is None:
            pbar = ProgBarCounter(max_samples)
        while n_samples < max_samples:
            if whole_paths:
                max_rollout_length = max_path_length
            else:
                max_rollout_length = min(max_path_length, max_samples - n_samples)
            path = rollout(
                _mdp,
                _policy,
                max_rollout_length,
                record_states
            )
            paths.append(path)
            n_new_samples = len(path["rewards"])
            n_samples += n_new_samples
            if queue is not None:
                queue.put(n_new_samples)
            else:
                pbar.inc(n_new_samples)
        if queue is None:
            pbar.stop()
        return paths
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def pool_init_theano(_):
    import os
    os.environ['THEANO_FLAGS'] = 'device=cpu'


def init_pool(n_parallel):
    global _n_parallel
    _n_parallel = n_parallel
    if _n_parallel > 1:
        global _pool
        _pool = MemmapingPool(_n_parallel)
        _pool.map(pool_init_theano, [None] * _n_parallel)


def reset():
    # pylint: disable=global-variable-not-assigned
    global _pool
    # pylint: enable=global-variable-not-assigned
    _pool.close()
    del _pool


def populate_task(mdp, policy):
    if _n_parallel > 1:
        _pool.map(pool_populate_task, [(mdp, policy)] * _n_parallel)
    else:
        pool_populate_task((mdp, policy))


def request_samples(
        policy_params,
        max_samples,
        max_path_length=np.inf,
        whole_paths=True,
        record_states=False):
    if _n_parallel > 1:
        manager = Manager()
        # pylint: disable=no-member
        queue = manager.Queue()
        # pylint: enable=no-member
        pool_max_samples = max_samples / _n_parallel
        args = dict(
            policy_params=policy_params,
            max_samples=pool_max_samples,
            max_path_length=max_path_length,
            queue=queue,
            whole_paths=whole_paths,
            record_states=record_states
        )
        paths_per_pool = _pool.map_async(pool_rollout, [args] * _n_parallel)
        pbar = ProgBarCounter(max_samples)
        while not paths_per_pool.ready():
            paths_per_pool.wait(0.1)
            while not queue.empty():
                pbar.inc(queue.get_nowait())
        pbar.stop()
        return sum(paths_per_pool.get(), [])
    else:
        args = dict(
            policy_params=policy_params,
            max_samples=max_samples,
            max_path_length=max_path_length,
            queue=None,
            whole_paths=whole_paths,
            record_states=record_states
        )
        return pool_rollout(args)
