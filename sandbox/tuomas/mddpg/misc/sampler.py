from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
from rllab.sampler.parallel_sampler import \
    _worker_set_policy_params, _worker_set_env_params, _worker_collect_one_path
import tensorflow as tf
import numpy as np
from distutils.version import LooseVersion


def worker_init_tf(G):
    G.sess = tf.Session()
    G.sess.__enter__()


def worker_init_tf_vars(G):
    if LooseVersion(tf.__version__) >= '0.12.1':
        # this suppresses annoying warning messages from tf
        initializer = tf.global_variables_initializer
    else:
        initializer = tf.initialize_all_variables
    G.sess.run(initializer())


def worker_init_head(G, k):
    G.policy.k = k


class ParallelSampler(BaseSampler):
    def start_worker(self, policy=None):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        if policy is None:
            policy = self.algo.policy
        parallel_sampler.populate_task(self.algo.env, policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    # TODO: needed?
    def collect_paths_for_heads(self, heads, max_path_length):
        args = [(h,) for h in heads]
        singleton_pool.run_each(
            worker_init_head,
            args,
        )
        results = singleton_pool.run_each(
            _worker_collect_one_path,
            [(max_path_length, None)] * singleton_pool.n_parallel
        )
        paths = [res[0] for res in results]
        return paths

    def collect_paths(self, max_path_length):
        results = singleton_pool.run_each(
            _worker_collect_one_path,
            [(max_path_length, None)] * singleton_pool.n_parallel
        )
        paths = [res[0] for res in results]
        return paths

    def obtain_samples(self, n_paths, max_path_length, policy=None):
        # copy latest params to the workers
        if policy is None:
            policy = self.algo.policy
        policy_params = policy.get_param_values()
        if hasattr(self.algo.env, "get_param_values"):
            env_params = self.algo.env.get_param_values()
        else:
            env_params = None
        scope = None

        singleton_pool.run_each(
            _worker_set_policy_params,
            [(policy_params, scope)] * singleton_pool.n_parallel
        )
        singleton_pool.run_each(
            _worker_set_env_params,
            [(env_params, scope)] * singleton_pool.n_parallel
        )

        paths = []
        for i in range(n_paths):
            paths += self.collect_paths(max_path_length)

        return paths
