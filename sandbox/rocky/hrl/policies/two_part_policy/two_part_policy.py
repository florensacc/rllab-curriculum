from __future__ import absolute_import
from __future__ import print_function

import theano.tensor as TT

from rllab.core.serializable import Serializable
from rllab.envs.base import EnvSpec
from rllab.policies.base import StochasticPolicy
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger
from sandbox.rocky.hrl.policies.two_part_policy.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.reflective_stochastic_mlp_policy import ReflectiveStochasticMLPPolicy


class TwoPartPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            env_spec,
            subgoal_dim,
            high_policy_cls,
            high_policy_args,
            low_policy_cls,
            low_policy_args,
            high_policy=None,
            low_policy=None,
            reparametrize_high_actions=None,
    ):
        """
        :param reparametrize_high_actions: whether to reparametrize high-level actions. If reparametrized,
        its probability distribution will not be included in computing the KL divergence and likelihood ratio terms
        :return:
        """
        Serializable.quick_init(self, locals())
        if high_policy_cls == CategoricalMLPPolicy or \
                                high_policy_cls == ReflectiveStochasticMLPPolicy and high_policy_args.get(
                    "action_policy_cls",
                    None) == CategoricalMLPPolicy:
            high_action_space = Discrete(subgoal_dim)
        else:
            high_action_space = Box(low=-1, high=1, shape=(subgoal_dim,))

        if reparametrize_high_actions is None:
            if high_policy_cls == CategoricalMLPPolicy or \
                                    high_policy_cls == ReflectiveStochasticMLPPolicy and high_policy_args.get(
                        "action_policy_cls",
                        None) == CategoricalMLPPolicy:
                reparametrize_high_actions = False
            else:
                reparametrize_high_actions = True

        high_env_spec = EnvSpec(
            observation_space=env_spec.observation_space,
            action_space=high_action_space,
        )
        low_env_spec = EnvSpec(
            observation_space=Product(env_spec.observation_space, high_action_space),
            action_space=env_spec.action_space,
        )
        if high_policy is None:
            high_policy = high_policy_cls(
                env_spec=high_env_spec,
                **high_policy_args
            )
        if low_policy is None:
            low_policy = low_policy_cls(
                env_spec=low_env_spec,
                **low_policy_args
            )
        self.high_policy = high_policy  # type: DeterministicMLPPolicy|GaussianMLPPolicy|ReflectiveStochasticMLPPolicy
        self.low_policy = low_policy  # type: GaussianMLPPolicy
        self.subgoal_dim = subgoal_dim
        self.high_policy_cls = high_policy_cls
        self.low_policy_cls = low_policy_cls
        self.high_policy_args = high_policy_args
        self.low_policy_args = low_policy_args
        self.high_env_spec = high_env_spec
        self.low_env_spec = low_env_spec
        self.reparametrize_high_actions = reparametrize_high_actions
        StochasticPolicy.__init__(self, env_spec=env_spec)

    def get_params_internal(self, **tags):
        return self.high_policy.get_params_internal(**tags) + self.low_policy.get_params_internal(**tags)

    def _split_dict(self, d):
        high_dict = dict()
        low_dict = dict()
        for k, v in d.iteritems():
            if k.startswith("high_"):
                high_dict[k[len("high_"):]] = v
            elif k.startswith("low_"):
                low_dict[k[len("low_"):]] = v
        return high_dict, low_dict

    def _merge_dict(self, high_dict, low_dict):
        d = dict()
        for k, v in high_dict.iteritems():
            d["high_%s" % k] = v
        for k, v in low_dict.iteritems():
            d["low_%s" % k] = v
        return d

    @property
    def recurrent(self):
        return self.high_policy.recurrent or self.low_policy.recurrent

    def action_dist_info_sym(self, obs_var, high_action_var):
        joint_obs_var = TT.concatenate([obs_var, high_action_var], axis=-1)
        return self.low_policy.dist_info_sym(joint_obs_var, dict())

    @property
    def action_dist(self):
        return self.low_policy.distribution

    def dist_info_sym(self, obs_var, state_info_vars):
        high_state_info_vars, low_state_info_vars = self._split_dict(state_info_vars)
        if self.recurrent:
            N, T, _ = obs_var.shape
        else:
            N, T = None, None
        if self.reparametrize_high_actions:
            high_obs = obs_var
            if self.low_policy.recurrent and not self.high_policy.recurrent:
                raise NotImplementedError
            high_action_sym = self.high_policy.get_reparam_action_sym(high_obs, high_state_info_vars)
            low_obs = TT.concatenate([obs_var, high_action_sym], axis=-1)
            if self.high_policy.recurrent and not self.low_policy.recurrent:
                # need to flatten the data before passing in
                low_obs = low_obs.reshape((N * T, -1))
                low_state_info_vars = {k: v.reshape((N * T, -1)) for k, v in low_state_info_vars.iteritems()}
            low_dist_info = self.low_policy.dist_info_sym(
                obs_var=low_obs,
                state_info_vars=low_state_info_vars
            )
            if self.high_policy.recurrent and not self.low_policy.recurrent:
                low_dist_info = {k: v.reshape((N, T, -1)) for k, v in low_dist_info.iteritems()}
            return self._merge_dict(dict(), low_dist_info)
        else:
            assert isinstance(self.high_policy, StochasticPolicy)
            # assert not self.recurrent
            high_dist_info = self.high_policy.dist_info_sym(
                obs_var=obs_var,
                state_info_vars=high_state_info_vars
            )
            high_action = state_info_vars["high_action"]
            low_dist_info = self.low_policy.dist_info_sym(
                obs_var=TT.concatenate([obs_var, high_action], axis=-1),
                state_info_vars=low_state_info_vars
            )
            return dict(self._merge_dict(high_dist_info, low_dist_info), high_action=high_action)

    def reset(self):
        self.high_policy.reset()
        self.low_policy.reset()

    def get_action(self, observation):
        high_action, high_agent_info = self.high_policy.get_action(observation)
        low_action, low_agent_info = self.low_policy.get_action((observation, high_action))
        return low_action, dict(
            self._merge_dict(high_agent_info, low_agent_info),
            high_action=self.high_env_spec.action_space.flatten(high_action)
        )

    @property
    def distribution(self):
        return self

    @property
    def dist_info_keys(self):
        high_dist = self.high_policy.distribution
        low_dist = self.low_policy.distribution
        return ["high_%s" % k for k in high_dist.dist_info_keys] + ["high_action"] + \
               ["low_%s" % k for k in low_dist.dist_info_keys]

    @property
    def state_info_keys(self):
        return ["high_%s" % k for k in self.high_policy.state_info_keys] + ["high_action"] + \
               ["low_%s" % k for k in self.low_policy.state_info_keys]

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_high, old_low = self._split_dict(old_dist_info_vars)
        new_high, new_low = self._split_dict(new_dist_info_vars)

        if self.reparametrize_high_actions:
            high_kl = 0.
        else:
            high_kl = self.high_policy.distribution.kl_sym(old_high, new_high)

        if self.high_policy.recurrent and not self.low_policy.recurrent:
            N, T, _ = old_dist_info_vars.values()[0].shape
            old_low = {k: v.reshape((N * T, -1)) for k, v in old_low.iteritems()}
            new_low = {k: v.reshape((N * T, -1)) for k, v in new_low.iteritems()}
            low_kl = self.low_policy.distribution.kl_sym(old_low, new_low)
            low_kl = low_kl.reshape((N, T))
        else:
            low_kl = self.low_policy.distribution.kl_sym(old_low, new_low)

        return high_kl + low_kl

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        old_high, old_low = self._split_dict(old_dist_info_vars)
        new_high, new_low = self._split_dict(new_dist_info_vars)
        if not self.reparametrize_high_actions:
            assert isinstance(self.high_policy, StochasticPolicy)
            high_action_var = old_dist_info_vars["high_action"]
            if not self.high_policy.recurrent and self.low_policy.recurrent:
                raise NotImplementedError
            high_lr = self.high_policy.distribution.likelihood_ratio_sym(high_action_var, old_high, new_high)
        else:
            high_lr = 1.

        if self.high_policy.recurrent and not self.low_policy.recurrent:
            N, T, _ = x_var.shape
            low_action_var = x_var.reshape((N * T, -1))
            old_low = {k: v.reshape((N * T, -1)) for k, v in old_low.iteritems()}
            new_low = {k: v.reshape((N * T, -1)) for k, v in new_low.iteritems()}
            low_lr = self.low_policy.distribution.likelihood_ratio_sym(low_action_var, old_low, new_low)
            low_lr = low_lr.reshape((N, T))
        else:
            low_lr = self.low_policy.distribution.likelihood_ratio_sym(x_var, old_low, new_low)

        return high_lr * low_lr

    def entropy(self, dist_info):
        # We only measure the entropy of the lower part. Otherwise it might be intractable in general
        _, low_dist = self._split_dict(dist_info)
        return self.low_policy.distribution.entropy(low_dist)

    def log_diagnostics(self, paths):
        high_paths = []
        low_paths = []
        for p in paths:
            high_agent_info, low_agent_info = self._split_dict(p["agent_infos"])
            high_paths.append(dict(p, agent_infos=high_agent_info))
            low_paths.append(dict(p, agent_infos=low_agent_info))
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
            self.high_policy.log_diagnostics(high_paths)
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
            self.low_policy.log_diagnostics(low_paths)


class DuelTwoPartPolicy(TwoPartPolicy):
    def __init__(self, env_spec, master_policy, share_gate=True):
        """
        :type master_policy: TwoPartPolicy
        :type env_spec: EnvSpec
        :param share_gate: whether to share the gate policy if the high-level policy of master_policy is recurrent
        """
        Serializable.quick_init(self, locals())
        if isinstance(master_policy.high_policy, ReflectiveStochasticMLPPolicy) and share_gate:
            # in this case, reuse the gate policy
            high_policy_args = dict(master_policy.high_policy_args, gate_policy=master_policy.high_policy.gate_policy)
        else:
            high_policy_args = master_policy.high_policy_args
        high_policy = master_policy.high_policy_cls(
            env_spec=master_policy.high_env_spec,
            **high_policy_args
        )
        low_policy = master_policy.low_policy
        TwoPartPolicy.__init__(
            self,
            env_spec=env_spec,
            subgoal_dim=master_policy.subgoal_dim,
            high_policy_cls=master_policy.high_policy_cls,
            high_policy_args=master_policy.high_policy_args,
            low_policy_cls=master_policy.low_policy_cls,
            low_policy_args=master_policy.low_policy_args,
            high_policy=high_policy,
            low_policy=low_policy,
            reparametrize_high_actions=master_policy.reparametrize_high_actions
        )
