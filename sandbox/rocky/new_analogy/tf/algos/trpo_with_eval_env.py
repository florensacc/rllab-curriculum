from rllab.misc import logger
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np


class TRPOWithEvalEnv(TRPO):
    def __init__(
            self,
            eval_env,
            eval_samples,
            eval_horizon,
            eval_frequency,
            **kwargs
    ):
        TRPO.__init__(self, **kwargs)
        self.eval_env = eval_env
        self.eval_samples = eval_samples
        self.eval_horizon = eval_horizon
        self.eval_frequency = eval_frequency
        n_vectorized_envs = min(100, max(1, int(np.ceil(eval_samples / eval_horizon))))
        self.eval_sampler = VectorizedSampler(
            env=eval_env, policy=self.policy, n_envs=n_vectorized_envs, parallel=False
        )
        self._itr = None

    def start_worker(self):
        TRPO.start_worker(self)
        self.eval_sampler.start_worker()

    def shutdown_worker(self):
        TRPO.shutdown_worker(self)
        self.eval_sampler.shutdown_worker()

    def obtain_samples(self, itr):
        self._itr = itr
        return TRPO.obtain_samples(self, itr)

    def log_diagnostics(self, paths):
        TRPO.log_diagnostics(self, paths)
        if self._itr % self.eval_frequency == 0:
            logger.log("Evaluating policy performance..")
            eval_paths = self.eval_sampler.obtain_samples(
                itr=self._itr, max_path_length=self.eval_horizon,
                batch_size=self.eval_samples)
            with logger.tabular_prefix('Eval|'), logger.prefix('Eval | '):
                self.eval_env.log_diagnostics(eval_paths)
                undiscounted_returns = [np.sum(path["rewards"]) for path in eval_paths]
                logger.record_tabular_misc_stat('Return', undiscounted_returns, placement='front')

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.eval_env,
        )
