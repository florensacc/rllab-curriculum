from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.npo import NPO

from rllab.core.serializable import Serializable
from sandbox.rocky.bayesian_nn_policy.policies.bayesian_nn_policy import BayesianNNPolicy
from sandbox.rocky.bayesian_nn_policy.optimizers.bnn_conjugate_gradient_optimizer import BNNConjugateGradientOptimizer
from sandbox.rocky.bayesian_nn_policy.optimizers.bnn_penalty_lbfgs_optimizer import BNNPenaltyLbfgsOptimizer
from rllab.sampler.stateful_pool import singleton_pool
from rllab.misc import tensor_utils
from rllab.misc import logger
import numpy as np


def _worker_set_policy_params(G, params):
    G.policy.set_param_values(params)


def rollout_ext(env, agent, max_path_length=np.inf):
    assert isinstance(agent, BayesianNNPolicy)
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    param_epsilon = agent.param_epsilon_var.get_value()
    param_val = agent.wrapped_policy.get_param_values()
    path_length = 0
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        param_epsilon=param_epsilon,
        param_val=param_val,
    )


def _worker_collect_one_path(G, max_path_length):
    path = rollout_ext(G.env, G.policy, max_path_length)
    return path, len(path["rewards"])


class BNNNPO(NPO):
    """
    BNN Trust Region Policy Optimization

    It builds on top of the TRPO implementation and add in the required tweaks.
    """

    def obtain_samples(self, itr):
        cur_params = self.policy.get_param_values()
        singleton_pool.run_each(
            _worker_set_policy_params,
            [(cur_params,)] * singleton_pool.n_parallel
        )
        return singleton_pool.run_collect(
            _worker_collect_one_path,
            threshold=self.batch_size,
            args=(self.max_path_length,),
            show_prog_bar=True
        )

    def optimize_policy(self, itr, samples_data):
        paths = samples_data["paths"]
        # param_mean = self.policy.mean_var.get_value()
        # param_log_std = self.policy.log_std_var.get_value()
        # for path in old_paths:
        #     path["param_epsilon"] = (path["param_val"] - param_mean) / np.exp(param_log_std)
        # paths = new_paths + old_paths
        all_advantages = np.concatenate([path["advantages"] for path in paths])
        adv_mean = np.mean(all_advantages)
        adv_std = np.std(all_advantages) + 1e-8
        for path in paths:
            path["advantages"] = (path["advantages"] - adv_mean) / adv_std
        # recompute epsilon for earlier paths
        loss_before = self.optimizer.loss(self.policy, paths)
        mean_kl_before = self.optimizer.constraint_val(self.policy, paths)
        self.optimizer.optimize(self.policy, paths)
        mean_kl = self.optimizer.constraint_val(self.policy, paths)
        loss_after = self.optimizer.loss(self.policy, paths)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        # rescale back the advantages
        for path in paths:
            path["advantages"] = path["advantages"] * adv_std + adv_mean
        return dict()


class BNNTRPO(BNNNPO, Serializable):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = BNNConjugateGradientOptimizer(**optimizer_args)
        super(BNNTRPO, self).__init__(optimizer=optimizer, **kwargs)


class BNNPPO(BNNNPO, Serializable):
    """
    Penalized Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = BNNPenaltyLbfgsOptimizer(**optimizer_args)
        super(BNNPPO, self).__init__(optimizer=optimizer, **kwargs)
