from joblib.pool import MemmapingPool
import os
from .utils import rollout, ProgBarCounter
from multiprocessing import Manager
import pyprind

__all__ = [
    'init_pool',
    'populate_task'
]

def pool_populate_task(args):
    global mdp, policy
    mdp, policy = args

def pool_rollout(args):
    global mdp, policy
    policy_params, max_samples, queue = args
    policy.set_param_values(policy_params)
    n_samples = 0
    paths = []
    if queue is None:
        pbar = ProgBarCounter(max_samples)
    while n_samples < max_samples:
        path = rollout(mdp, policy, max_samples - n_samples)
        paths.append(path)
        n_new_samples = len(path["states"])
        n_samples += n_new_samples
        if queue is not None:
            queue.put(n_new_samples)
        else:
            pbar.inc(n_new_samples)
    if queue is None:
        pbar.stop()
    return paths

def pool_init_theano(_):
    import os
    os.environ['THEANO_FLAGS'] = 'device=cpu'

def init_pool(_n_parallel):
    global n_parallel
    n_parallel = _n_parallel
    if n_parallel > 1:
        global pool
        pool = MemmapingPool(_n_parallel)
        pool.map(pool_init_theano, [None] * n_parallel)

def populate_task(mdp, policy):
    global n_parallel
    if n_parallel > 1:
        global pool
        pool.map(pool_populate_task, [(mdp, policy)] * n_parallel)
    else:
        pool_populate_task((mdp, policy))

def request_samples(policy_params, max_samples):
    global n_parallel
    if n_parallel > 1:
        manager = Manager()
        queue = manager.Queue()
        pool_max_samples = max_samples / n_parallel
        paths_per_pool = pool.map_async(pool_rollout, [(policy_params, pool_max_samples, queue)] * n_parallel)
        pbar = ProgBarCounter(max_samples)
        while not paths_per_pool.ready():
            paths_per_pool.wait(0.1)
            while not queue.empty():
                pbar.inc(queue.get_nowait())
        pbar.stop()
        return sum(paths_per_pool.get(), [])
    else:
        return pool_rollout((policy_params, max_samples, None))
