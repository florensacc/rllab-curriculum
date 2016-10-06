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

        # with tf.variable_scope(name):
        summary_network, action_network = net.new_networks(env_spec=env_spec)

        self.summary_network = summary_network
        self.action_network = action_network

        self.f_update_summary = tensor_utils.compile_function(
            summary_network.input_vars,
            summary_network.get_update_op(phase='test'),
        )

        obs_var = env_spec.observation_space.new_tensor_variable(
            "obs",
            extra_dims=1
        )
        # summary_var = tf.placeholder(tf.float32, (None, summary_network.output_dim), "summary")
        dones_var = tf.placeholder(tf.float32, (None,), "dones")

        action_inputs = [obs_var]#, summary_var]
        action_input_args = dict(obs_var=obs_var)#, summary_var=summary_var)

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

        # LayersPowered.__init__(self, [summary_network.output_layer, action_network.output_layer])

    def get_params_internal(self, **tags):
        return sorted(
            set(self.summary_network.get_params(**tags) + self.action_network.get_params(**tags)),
            key=lambda x: x.name
        )

    @property
    def action_dim(self):
        return self.action_space.flat_dim

    def action_sym(self, obs_var, state_info_vars, **kwargs):
        demo_obs_var = state_info_vars["demo_obs"]
        demo_action_var = state_info_vars["demo_action"]

        summary_var = self.summary_network.get_output(
            obs_var=demo_obs_var,
            action_var=demo_action_var,
            **kwargs
        )

        return self.action_network.get_output(
            obs_var=obs_var,
            summary_var=summary_var,
            # state_info_vars=state_info_vars,
            **kwargs
        )

    def apply_demos(self, paths):

        max_len = np.max([len(p["rewards"]) for p in paths])

        demo_obs = [p["observations"] for p in paths]
        demo_obs = np.asarray([tensor_utils.pad_tensor(o, max_len) for o in demo_obs])

        demo_actions = [p["actions"] for p in paths]
        demo_actions = np.asarray([tensor_utils.pad_tensor(a, max_len) for a in demo_actions])

        demo_valids = [np.ones_like(p["rewards"]) for p in paths]
        demo_valids = np.asarray([tensor_utils.pad_tensor(v, max_len) for v in demo_valids])

        assert np.all(demo_valids)

        self.f_update_summary(demo_obs, demo_actions)

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        actions = self.f_action(flat_obs)
        # agent_info = dict(zip(self.action_network.state_info_keys, state_info_list))
        # print("prev action:", np.linalg.norm(self.action_network.prev_action_var.eval()))
        # print("prev state:", np.linalg.norm(self.action_network.prev_state_var.eval()))
        return actions, dict()#agent_info

    @property
    def recurrent(self):
        return self.action_network.recurrent

    # @property
    # def state_info_specs(self):
    #     return self.action_network.state_info_specs

    def reset(self, dones=None):
        if self.action_network.recurrent:
            if dones is None:
                dones = [True]
            if len(dones) != self.prev_reset_length:
                self.f_full_reset(dones)
            else:
                self.f_partial_reset(dones)
            self.prev_reset_length = len(dones)
            # if np.any(dones):
            #     print("summary:", np.linalg.norm(self.summary_network.summary_var.eval()[dones]))
            #     print("prev action:", np.linalg.norm(self.action_network.prev_action_var.eval()[dones]))
            #     print("prev state:", np.linalg.norm(self.action_network.prev_state_var.eval()[dones]))
