import random

from joblib.pool import MemmapingPool
from rllab.sampler.utils import rollout, ProgBarCounter
from rllab.misc.ext import extract, set_seed
from multiprocessing import Manager, Queue
import numpy as np
import traceback
import sys

__all__ = [
    'config_parallel_sampler',
    'populate_task',
    'reset',
]


class Globals(object):

    def __init__(self):
        self.mdp = None
        self.policy = None
        self.n_parallel = 1
        self.pool = None
        self.base_seed = 0
        # This queue is used to ensure that each worker is properly initialized
        self.queue = None
        self.paths = []
        # This queue is used by each worker to communicate to the master that
        # it has been initialized
        self.worker_queue = None


G = Globals()


def pool_init_theano():
    import os
    os.environ['THEANO_FLAGS'] = 'device=cpu'


def processor_init(queue):
    pool_init_theano()
    args = queue.get()
    worker_init(*args)


def worker_init(mdp, policy, seed_inc):
    set_seed(seed_inc + G.base_seed)
    pool_init_theano()
    G.mdp, G.policy = mdp, policy
    if G.worker_queue:
        G.worker_queue.put(None)


def pool_rollout(args):
    try:
        policy_params, max_samples, max_path_length, queue, record_states, whole_paths = \
            extract(
                args,
                "policy_params", "max_samples", "max_path_length", "queue",
                "record_states", "whole_paths"
            )
        G.policy.set_param_values(policy_params)
        n_samples = 0
        G.paths = []
        if queue is None:
            pbar = ProgBarCounter(max_samples)
        while n_samples < max_samples:
            if whole_paths:
                max_rollout_length = max_path_length
            else:
                max_rollout_length = min(
                    max_path_length, max_samples - n_samples)
            path = rollout(
                G.mdp,
                G.policy,
                max_rollout_length,
                record_states
            )
            G.paths.append(path)
            n_new_samples = len(path["rewards"])
            n_samples += n_new_samples
            if queue is not None:
                queue.put(n_new_samples)
            else:
                pbar.inc(n_new_samples)
        if queue is None:
            pbar.stop()
        return len(G.paths)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def config_parallel_sampler(n_parallel, base_seed):
    G.n_parallel = n_parallel

    if G.n_parallel > 1:
        G.base_seed = base_seed if base_seed else random.randint()
        m = Manager()
        G.queue = m.Queue()
        G.worker_queue = m.Queue()

        G.pool = MemmapingPool(
            G.n_parallel,
            initializer=processor_init,
            initargs=[G.queue]
        )


def reset():
    G.pool.close()
    G.pool = None


def populate_task(mdp, policy):
    if G.n_parallel > 1:
        # pipes = []
        for i in xrange(G.n_parallel):
            G.queue.put((mdp, policy, i))
        print "Waiting for all workers to be initialized"
        for i in xrange(G.n_parallel):
            G.worker_queue.get()
        print "all workers initialized"
    else:
        worker_init(mdp, policy, 0)


def worker_run_task(all_args):
    runner, args, kwargs = all_args
    # signals to the master that this task is up and running
    G.worker_queue.put(None)
    # wait for the master to signal continuation
    G.queue.get()
    return runner(*args, **kwargs)


def run_map(runner, *args, **kwargs):
    if G.n_parallel > 1:
        results = G.pool.map_async(
            worker_run_task, [(runner, args, kwargs)] * G.n_parallel)
        for i in range(G.n_parallel):
            G.worker_queue.get()
        for i in range(G.n_parallel):
            G.queue.put(None)
        return results.get()
    return [runner(*args, **kwargs)]


def master_collect_mean(worker_f, *args, **kwargs):
    try:
        results = run_map(worker_f, *args, **kwargs)
        if isinstance(results[0], (list, tuple)):
            # returning a list / tuple of results, compute the mean of each of them
            n_results = len(results[0])
            ret = [np.mean(np.array([np.array(x[i]) for x in results]), axis=0)
                   for i in range(n_results)]
            if isinstance(results[0], tuple):
                ret = tuple(ret)
            return ret
        else:
            return np.mean(np.array(results), axis=0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        import ipdb; ipdb.set_trace()


def worker_set_param_values(params, **tags):
    G.policy.set_param_values(params, **tags)


def master_set_param_values(params, **tags):
    run_map(worker_set_param_values, params, **tags)


def request_samples(
        policy_params,
        max_samples,
        max_path_length=np.inf,
        whole_paths=True,
        record_states=False):
    if G.n_parallel > 1:
        manager = Manager()
        # pylint: disable=no-member
        queue = manager.Queue()
        # pylint: enable=no-member
        pool_max_samples = max_samples / G.n_parallel
        args = dict(
            policy_params=policy_params,
            max_samples=pool_max_samples,
            max_path_length=max_path_length,
            queue=queue,
            whole_paths=whole_paths,
            record_states=record_states
        )
        paths_per_pool = G.pool.map_async(pool_rollout, [args] * G.n_parallel)
        pbar = ProgBarCounter(max_samples)
        while not paths_per_pool.ready():
            paths_per_pool.wait(0.1)
            while not queue.empty():
                pbar.inc(queue.get_nowait())
        pbar.stop()
        paths_per_pool.get()
        # sanity check
        # print [[p1['states'].shape == p2['states'].shape and \
        #   np.allclose(p1['states'], p2['states']) \
        #     for p1,p2 in zip(paths1, paths2)] for paths1, paths2 in \
        #      zip(ps[1:], ps[:-1])]
    else:
        args = dict(
            policy_params=policy_params,
            max_samples=max_samples,
            max_path_length=max_path_length,
            queue=None,
            whole_paths=whole_paths,
            record_states=record_states
        )
        pool_rollout(args)
