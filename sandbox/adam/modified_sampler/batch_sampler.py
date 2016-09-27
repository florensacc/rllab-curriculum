"""
Differences from normal BatchSampler class:
1. imports modified parallel_sampler
"""


# from rllab.sampler import parallel_sampler
from rllab.sampler.base import BaseSampler

from sandbox.adam.modified_sampler import parallel_sampler


class BatchSampler(BaseSampler):

    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated
