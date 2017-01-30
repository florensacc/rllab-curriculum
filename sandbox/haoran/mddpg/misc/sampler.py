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

class MNNParallelSampler(BaseSampler):
    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.algo.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

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

    def obtain_samples(self, itr, max_path_length, max_head_repeat):
        # copy latest params to the workers
        policy_params = self.algo.policy.get_param_values()
        if hasattr(self.algo.env,"get_param_values"):
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

        # set heads for different workers and then sample in parallel
        # some heads may be sampled more than once
        # discard repeated paths from the same heads
        # this is reasonable if the env is deterministic
        paths = []
        K = self.algo.policy.K
        n = singleton_pool.n_parallel
        first_head = 0
        head_repeat_count = np.zeros(K, dtype=int)
        while np.amin(head_repeat_count) < max_head_repeat:
            heads = list(range(first_head, first_head + n))
            heads = np.mod(heads, K)
            cur_paths = self.collect_paths_for_heads(heads, max_path_length)
            cur_heads = [path["agent_infos"]["heads"][0] for path in cur_paths]
            for path, head in zip(cur_paths, cur_heads):
                if head_repeat_count[head] < max_head_repeat:
                    paths.append(path)
                    head_repeat_count[head] += 1
            first_head += n
        # debugging: print the sampled heads
        # print([path["agent_infos"]["heads"][0] for path in paths])

        # truncate paths to maintain a consistent number of samples per iter
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated
