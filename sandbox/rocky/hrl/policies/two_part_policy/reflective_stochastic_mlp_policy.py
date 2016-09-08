


import lasagne.layers as L
import lasagne.nonlinearities as NL
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.core.network import MLP
from rllab.core.parameterized import Parameterized
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.misc import ext
from rllab.policies.base import StochasticPolicy
from rllab.envs.base import EnvSpec
from rllab.spaces.product import Product
from sandbox.rocky.hrl.policies.two_part_policy.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.deterministic_mlp_policy import DeterministicMLPPolicy
import theano
import theano.tensor as TT
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


floatX = theano.config.floatX


class ReflectiveStochasticMLPPolicy(StochasticPolicy, Parameterized, Serializable):
    """
    This is intended to be used as a building block for the high-level policy.
    """

    def __init__(
            self,
            env_spec,
            action_policy=None,
            action_policy_cls=None,
            action_policy_args=None,
            gate_policy=None,
            gate_policy_cls=None,
            gate_policy_args=None,
            gated=True,
            init_state=None,
            init_state_trainable=None,  # True,
            truncate_gradient=-1,
    ):
        Serializable.quick_init(self, locals())
        # assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        if action_policy is None:
            if action_policy_cls is None:
                action_policy_cls = GaussianMLPPolicy
            if action_policy_args is None:
                action_policy_args = dict()
            action_policy = action_policy_cls(
                env_spec=EnvSpec(
                    observation_space=Product(env_spec.observation_space, env_spec.action_space),
                    action_space=env_spec.action_space,
                ),
                **action_policy_args
            )

        if gated:
            if gate_policy is None:
                if gate_policy_cls is None:
                    gate_policy_cls = GaussianMLPPolicy
                if gate_policy_args is None:
                    gate_policy_args = dict()
                gate_policy = gate_policy_cls(
                    env_spec=EnvSpec(
                        observation_space=Product(env_spec.observation_space, env_spec.action_space),
                        action_space=Box(low=0, high=1, shape=(1,))
                    ),
                    **gate_policy_args
                )
        else:
            gate_policy = None

        if isinstance(action_policy, CategoricalMLPPolicy) and gate_policy is not None:
            assert isinstance(gate_policy, DeterministicMLPPolicy)
        if gate_policy is not None:
            assert gate_policy.output_nonlinearity is None

        if init_state_trainable is None:
            # set default value based on type of action space
            if isinstance(env_spec.action_space, Discrete):
                init_state_trainable = False
            else:
                init_state_trainable = True

        if init_state is None:
            if isinstance(env_spec.action_space, Discrete):
                init_state = np.eye(action_dim)[0]
                assert not init_state_trainable
            else:
                init_state = np.random.uniform(low=-1, high=1, size=(action_dim,))
        self.init_state_var = theano.shared(np.cast[floatX](init_state), name="init_state")
        self.init_state_trainable = init_state_trainable
        self.prev_action = None
        self.action_policy = action_policy
        self.gate_policy = gate_policy
        self.truncate_gradient = truncate_gradient

        super(ReflectiveStochasticMLPPolicy, self).__init__(env_spec)

    def reset(self):
        self.prev_action = self.action_space.unflatten(self.init_state_var.get_value())
        self.action_policy.reset()
        if self.gate_policy is not None:
            self.gate_policy.reset()

    def _merge_dict(self, action_dict, gate_dict):
        d = dict()
        for k, v in action_dict.items():
            d["action_%s" % k] = v
        for k, v in gate_dict.items():
            d["gate_%s" % k] = v
        return d

    def _split_dict(self, d):
        action_dict = dict()
        gate_dict = dict()
        for k, v in d.items():
            if k.startswith("action_"):
                action_dict[k[len("action_"):]] = v
            elif k.startswith("gate_"):
                gate_dict[k[len("gate_"):]] = v
        return action_dict, gate_dict

    def get_action(self, observation):
        if isinstance(self.action_policy, CategoricalMLPPolicy):
            # Special treatment
            action, action_info = self.action_policy.get_action((observation, self.prev_action))
            prev_action_repr = self.action_space.flatten(self.prev_action)
            if self.gate_policy is not None:
                gate, gate_info = self.gate_policy.get_action((observation, self.prev_action))
                gate = sigmoid(gate)
                action_prob = action_info["prob"] * gate + prev_action_repr * (1 - gate)
                if np.random.uniform() > gate:
                    action = self.prev_action
                action_info = dict(prob=action_prob)
            action_info = dict(action_info, prev_action=prev_action_repr)
            self.prev_action = action
            return action, action_info
        else:
            action, action_info = self.action_policy.get_action((observation, self.prev_action))
            if self.gate_policy is not None:
                gate, gate_info = self.gate_policy.get_action((observation, self.prev_action))
                gate = sigmoid(gate)
                action = action * gate + self.prev_action * (1 - gate)
            else:
                gate_info = dict()
            self.prev_action = action
            return action, self._merge_dict(action_info, gate_info)

    @property
    def recurrent(self):
        # if discrete, stochastic actions, can treat as non-recurrent since we can't reparametrize anyways
        if isinstance(self.action_policy, CategoricalMLPPolicy):
            return False
        return True

    def get_reparam_action_sym(self, obs_var, state_info_vars):
        assert not isinstance(self.action_policy, CategoricalMLPPolicy)
        # obs_var: N * T * S
        action_info, gate_info = self._split_dict(state_info_vars)
        N = obs_var.shape[0]
        init_state = TT.tile(TT.reshape(self.init_state_var, (1, -1)), (N, 1))

        action_info_list = [action_info[k] for k in self.action_policy.state_info_keys]
        if self.gate_policy is not None:
            gate_info_list = [gate_info[k] for k in self.gate_policy.state_info_keys]
        else:
            gate_info_list = []

        action_info_seq = [x.dimshuffle(1, 0, 2) for x in action_info_list]
        gate_info_seq = [x.dimshuffle(1, 0, 2) for x in gate_info_list]

        def step(cur_obs, *args):
            cur_action_info_list = args[:len(action_info_seq)]
            cur_gate_info_list = args[len(action_info_seq):len(action_info_seq) + len(gate_info_seq)]
            prev_action = args[-1]

            cur_action_info = {k: v for k, v in zip(self.action_policy.state_info_keys, cur_action_info_list)}
            if self.gate_policy is not None:
                cur_gate_info = {k: v for k, v in zip(self.gate_policy.state_info_keys, cur_gate_info_list)}
            else:
                cur_gate_info = dict()

            joint_obs = TT.concatenate([cur_obs, prev_action], axis=-1)

            action_sym = self.action_policy.get_reparam_action_sym(joint_obs, cur_action_info)
            if self.gate_policy is not None:
                gate_sym = self.gate_policy.get_reparam_action_sym(joint_obs, cur_gate_info)
                gate_sym = TT.nnet.sigmoid(gate_sym)
                gate_sym = TT.tile(gate_sym, [1] * (gate_sym.ndim - 1) + [action_sym.shape[-1]])
                action_sym = action_sym * gate_sym + prev_action * (1 - gate_sym)
            return action_sym

        permuted_actions, _ = theano.scan(
            step,
            sequences=[obs_var.dimshuffle(1, 0, 2)] + action_info_seq + gate_info_seq,
            outputs_info=init_state,
            truncate_gradient=self.truncate_gradient,
        )
        return permuted_actions.dimshuffle(1, 0, 2)

    @property
    def distribution(self):
        if isinstance(self.action_policy, CategoricalMLPPolicy):
            return self.action_policy.distribution
        return self

    @property
    def dist_info_keys(self):
        keys = ["action_%s" % k for k in self.action_policy.distribution.dist_info_keys]
        if self.gate_policy is not None:
            keys += ["gate_%s" % k for k in self.gate_policy.distribution.dist_info_keys]
        return keys

    @property
    def state_info_keys(self):
        if isinstance(self.action_policy, CategoricalMLPPolicy):
            return self.action_policy.state_info_keys + ["prev_action"]
        keys = ["action_%s" % k for k in self.action_policy.state_info_keys]
        if self.gate_policy is not None:
            keys += ["gate_%s" % k for k in self.gate_policy.state_info_keys]
        return keys

    def get_params_internal(self, **tags):
        params = self.action_policy.get_params_internal(**tags)
        if self.gate_policy is not None:
            params += self.gate_policy.get_params_internal(**tags)
        if self.init_state_trainable or tags.get("trainable", False):
            params = params + [self.init_state_var]
        return params

    def dist_info(self, obs, state_infos):
        assert isinstance(self.action_policy, CategoricalMLPPolicy)
        prev_actions = state_infos["prev_action"]
        joint_obs = np.concatenate([obs, prev_actions], axis=-1)
        dist_info = self.action_policy.dist_info(joint_obs, state_infos)
        if self.gate_policy is not None:
            assert isinstance(self.gate_policy, DeterministicMLPPolicy)
            action_prob = dist_info["prob"]
            gates = sigmoid(self.gate_policy.f_action(joint_obs))
            action_prob = action_prob * gates + prev_actions * (1 - gates)
            dist_info = dict(prob=action_prob)
        return dist_info

    def dist_info_sym(self, obs_var, state_info_vars):
        assert isinstance(self.action_policy, CategoricalMLPPolicy)
        # action_state_info_vars, _ = self._split_dict(state_info_vars)
        prev_action_var = state_info_vars["prev_action"]
        joint_obs_var = TT.concatenate([obs_var, prev_action_var], axis=-1)
        dist_info = self.action_policy.dist_info_sym(joint_obs_var, state_info_vars)
        if self.gate_policy is not None:
            assert isinstance(self.gate_policy, DeterministicMLPPolicy)
            action_prob = dist_info["prob"]
            gate_var = TT.nnet.sigmoid(self.gate_policy.get_reparam_action_sym(joint_obs_var, state_info_vars))
            gate_var = TT.tile(gate_var, [1] * (gate_var.ndim - 1) + [action_prob.shape[-1]])
            action_prob = action_prob * gate_var + prev_action_var * (1 - gate_var)
            dist_info = dict(prob=action_prob)
        return dist_info
