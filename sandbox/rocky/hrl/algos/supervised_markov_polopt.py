from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from rllab.policies.base import StochasticPolicy
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from sandbox.rocky.hrl.envs.supervised_env import SupervisedEnv
from rllab.sampler import parallel_sampler
from rllab.misc import logger
import theano.tensor as TT
import numpy as np


class SupervisedMarkovPolopt(RLAlgorithm):
    def __init__(self,
                 env,
                 policy,
                 optimizer=None,
                 discount=0.99,
                 n_itr=100,
                 n_test_samples=10000,
                 max_path_length=100):
        """
        :type env: SupervisedEnv
        :type policy: StochasticPolicy
        :type optimizer: LbfgsOptimizer
        """
        self.env = env
        self.policy = policy
        if optimizer is None:
            optimizer = LbfgsOptimizer(max_opt_itr=20)
        self.optimizer = optimizer
        self.discount = discount
        self.n_itr = n_itr
        self.n_test_samples = n_test_samples
        self.max_path_length = max_path_length

    def train(self):
        training_paths = self.env.generate_training_paths()
        self.env.test_mode()
        parallel_sampler.populate_task(self.env, self.policy)
        obs_var = self.env.observation_space.new_tensor_variable(
            name="obs",
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            name="action",
            extra_dims=1
        )
        dist_info_sym = self.policy.dist_info_sym(obs_var, dict())

        neg_logli = -TT.mean(self.policy.distribution.log_likelihood_sym(action_var, dist_info_sym))

        self.optimizer.update_opt(neg_logli, self.policy, [obs_var, action_var])

        all_observations = np.concatenate([p["observations"] for p in training_paths])
        all_actions = np.concatenate([p["actions"] for p in training_paths])

        for itr in range(self.n_itr):
            self.optimizer.optimize([all_observations, all_actions])
            paths = parallel_sampler.sample_paths(
                policy_params=self.policy.get_param_values(),
                max_samples=self.n_test_samples,
                max_path_length=self.max_path_length
            )
            avg_discounted_return = np.mean([
                np.sum((self.discount ** np.arange(len(p["rewards"]))) * p["rewards"])
                for p in paths
            ])
            avg_return = np.mean([
                np.sum(p["rewards"])
                for p in paths
            ])
            logger.record_tabular("Itr", itr)
            logger.record_tabular("AverageDiscountedReturn", avg_discounted_return)
            logger.record_tabular("AverageReturn", avg_return)
            logger.dump_tabular()
