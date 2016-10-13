from rllab.core.serializable import Serializable
from sandbox.rocky.analogy.policies.base import AnalogyPolicy
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.misc import tensor_utils
import numpy as np
import tensorflow as tf


class ModularAnalogyPolicy(AnalogyPolicy, Parameterized, Serializable):
    def __init__(self, env_spec, name, net):
        Serializable.quick_init(self, locals())

        AnalogyPolicy.__init__(self, env_spec=env_spec)

        summary_network, action_network = net.new_networks(env_spec=env_spec)

        self.summary_network = summary_network
        self.action_network = action_network

        summary_obs_var = env_spec.observation_space.new_tensor_variable(
            "summary_obs",
            extra_dims=2
        )
        summary_actions_var = env_spec.action_space.new_tensor_variable(
            "summary_actions",
            extra_dims=2
        )
        summary_valids_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="summary_valids")

        self.f_update_summary = tensor_utils.compile_function(
            [summary_obs_var, summary_actions_var, summary_valids_var],
            summary_network.get_update_op(
                obs_var=summary_obs_var,
                actions_var=summary_actions_var,
                valids_var=summary_valids_var,
                phase='test'
            ),
        )

        action_obs_var = env_spec.observation_space.new_tensor_variable(
            "action_obs",
            extra_dims=1
        )
        dones_var = tf.placeholder(tf.float32, (None,), "dones")

        action_inputs = [action_obs_var, summary_valids_var]
        action_input_args = dict(obs_var=action_obs_var, demo_valids_var=summary_valids_var)

        if action_network.recurrent:
            prev_action_var = env_spec.action_space.new_tensor_variable(
                "prev_action",
                extra_dims=1,
            )
            prev_state_var = tf.placeholder(tf.float32, (None, action_network.state_dim), "prev_state")
            action_inputs.append(prev_action_var)
            action_inputs.append(prev_state_var)
            action_input_args["prev_action_var"] = prev_action_var
            action_input_args["prev_state_var"] = prev_state_var

        self.f_action = tensor_utils.compile_function(
            action_inputs,
            action_network.get_step_op(
                **action_input_args,
                phase='test'
            )
        )

        self.f_full_reset = tensor_utils.compile_function(
            [dones_var],
            action_network.get_full_reset_op(dones_var),
        )
        self.f_partial_reset = tensor_utils.compile_function(
            [dones_var],
            action_network.get_partial_reset_op(dones_var),
        )

        self.prev_reset_length = -1

        self.summary_network = summary_network
        self.action_network = action_network

    def get_params_internal(self, **tags):
        params = sorted(
            set(self.summary_network.get_params(**tags) + self.action_network.get_params(**tags)),
            key=lambda x: x.name
        )
        return params

    @property
    def action_dim(self):
        return self.action_space.flat_dim

    def action_sym(self, obs_var, state_info_vars, **kwargs):
        demo_obs_var = state_info_vars["demo_obs"]
        demo_actions_var = state_info_vars["demo_actions"]
        demo_valids_var = state_info_vars["demo_valids"]

        summary_var = self.summary_network.get_output(
            obs_var=demo_obs_var,
            actions_var=demo_actions_var,
            valids_var=demo_valids_var,
            **kwargs
        )

        return self.action_network.get_output(
            obs_var=obs_var,
            summary_var=summary_var,
            demo_valids_var=demo_valids_var,
            **kwargs
        )

    def apply_demo(self, path):
        self.apply_demos([path])

    def apply_demos(self, paths):

        max_len = np.max([len(p["observations"]) for p in paths])

        demo_obs = [p["observations"] for p in paths]
        demo_obs = tensor_utils.pad_tensor_n(demo_obs, max_len)

        demo_actions = [p["actions"] for p in paths]
        demo_actions = tensor_utils.pad_tensor_n(demo_actions, max_len)

        demo_valids = [np.ones(len(p["observations"])) for p in paths]
        demo_valids = tensor_utils.pad_tensor_n(demo_valids, max_len)

        self.demo_valids = demo_valids
        self.f_update_summary(demo_obs, demo_actions, demo_valids)

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        actions = self.f_action(flat_obs, self.demo_valids)
        return actions, dict()

    @property
    def recurrent(self):
        return self.action_network.recurrent

    def reset(self, dones=None):
        if self.action_network.recurrent:
            if dones is None:
                dones = [True]
            if len(dones) != self.prev_reset_length:
                self.f_full_reset(dones)
            else:
                self.f_partial_reset(dones)
            self.prev_reset_length = len(dones)
