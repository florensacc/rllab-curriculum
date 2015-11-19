from joblib.pool import MemmapingPool
import os
from rllab.sampler.utils import rollout, ProgBarCounter
from rllab.misc.ext import extract
from rllab.misc.special import explained_variance_1d, discount_cumsum
from multiprocessing import Manager
import pyprind
import numpy as np
from rllab.misc import logger

__all__ = [
    'init_pool',
    'populate_task',
    'reset',
]

def pool_populate_task(args):
    global mdp, policy
    mdp, policy = args

def pool_rollout(args):
    global mdp, policy
    policy_params, max_samples, max_path_length, queue, record_states = \
        extract(
            args,
            "policy_params", "max_samples", "max_path_length", "queue", "record_states"
        )
    prev_params = policy.get_param_values()
    policy.set_param_values(policy_params)
    n_samples = 0
    paths = []
    if queue is None:
        pbar = ProgBarCounter(max_samples)
    while n_samples < max_samples:
        path = rollout(mdp, policy, min(max_path_length, max_samples - n_samples), record_states)
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

def reset():
    if 'pool' in globals():
        global pool
        pool.close()
        del pool

def _get_n_parallel():
    global n_parallel
    if 'n_parallel' not in globals():
        logger.log('parallel_sampler not initialized. default to single-threaded sampling')
        n_parallel = 1
    return n_parallel


def populate_task(mdp, policy):
    if _get_n_parallel() > 1:
        global pool
        pool.map(pool_populate_task, [(mdp, policy)] * n_parallel)
    else:
        pool_populate_task((mdp, policy))

def request_samples(policy_params, max_samples, max_path_length=np.inf, record_states=False):
    if _get_n_parallel() > 1:
        manager = Manager()
        queue = manager.Queue()
        pool_max_samples = max_samples / n_parallel
        args = dict(
            policy_params=policy_params,
            max_samples=pool_max_samples,
            max_path_length=max_path_length,
            queue=queue,
            record_states=record_states
        )
        paths_per_pool = pool.map_async(pool_rollout, [args] * _get_n_parallel())
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
            record_states=record_states
        )
        return pool_rollout(args)

def request_samples_stats(itr, mdp, policy, vf, samples_per_itr, max_path_length, discount=1, gae_lambda=1, record_states=False):
    """
    Perform rollout to obtain samples according to the given parameters. It
    returns a few quantities that may be useful for the algorithm:

    all_obs: all observations concatenated together
    """

    cur_params = policy.get_param_values()
    paths = request_samples(
        policy_params=cur_params,
        max_samples=samples_per_itr,
        max_path_length=max_path_length,
        record_states=record_states
    )

    all_baselines = []
    all_returns = []

    for path in paths:
        path["returns"] = discount_cumsum(path["rewards"], discount)
        baselines = np.append(vf.predict(path), 0)
        deltas = path["rewards"] + discount*baselines[1:] - baselines[:-1]
        path["advantage"] = discount_cumsum(deltas, discount*gae_lambda)
        all_baselines.append(baselines[:-1])
        all_returns.append(path["returns"])

    ev = explained_variance_1d(np.concatenate(all_baselines), np.concatenate(all_returns))

    all_obs = np.vstack([path["observations"] for path in paths])
    all_states = np.vstack([path["states"] for path in paths])
    all_pdists = np.vstack([path["pdists"] for path in paths])
    all_actions = np.vstack([path["actions"] for path in paths])
    all_advantages = np.concatenate([path["advantage"] for path in paths])

    avg_return = np.mean([sum(path["rewards"]) for path in paths])

    ent = policy.compute_entropy(all_pdists)

    logger.record_tabular('Iteration', itr)
    logger.record_tabular('Entropy', ent)
    logger.record_tabular('Perplexity', np.exp(ent))
    logger.record_tabular('AvgReturn', avg_return)
    logger.record_tabular('NumTrajs', len(paths))
    logger.record_tabular('ExplainedVariance', ev)

    return dict(
        all_obs=all_obs,
        all_advantages=all_advantages,
        all_actions=all_actions,
        all_pdists=all_pdists,
        all_states=all_states,
        paths=paths,
    )
