from rllab.core.serializable import Serializable
from rllab.misc import logger
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
import tensorflow as tf

from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.core.parameterized import JointParameterized
from sandbox.rocky.tf.misc import tensor_utils
import numpy as np


class SharedWeightsBaseline(LayersPowered, Serializable):
    def __init__(self, env_spec, policy, hidden_nonlinearity=tf.nn.relu, hidden_sizes=(32, 32)):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.policy = policy

        # We extract features from this layer
        rnn_network = policy.prob_network
        l_rnn = rnn_network.recurrent_layer
        l_flat_rnn = L.ReshapeLayer(
            l_rnn, shape=(-1, rnn_network.hidden_dim),
            name="gru_flat"
        )

        value_network = MLP(
            input_shape=(rnn_network.hidden_dim,),
            input_layer=l_flat_rnn,
            hidden_nonlinearity=tf.nn.relu,
            hidden_sizes=(32, 32),
            output_dim=1,
            output_nonlinearity=None,
            name="value_network"
        )

        self.value_network = value_network
        self.l_value = value_network.output_layer
        self.l_input = policy.l_input
        self.l_rnn = l_rnn

        LayersPowered.__init__(self, value_network.output_layer)

        obs_var = env_spec.observation_space.new_tensor_variable("obs", extra_dims=2)
        state_info_vars = {
            k: tf.placeholder(tf.float32, (None, None) + shape, k) for k, shape in policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k, _ in policy.state_info_specs]

        self.f_predict = tensor_utils.compile_function(
            inputs=[obs_var] + state_info_vars_list,
            outputs=self.predict_sym(obs_var, state_info_vars)
        )

    def predict_sym(self, obs_var, state_info_vars, **kwargs):
        obs_var = tf.cast(obs_var, tf.float32)
        if self.policy.state_include_action:
            prev_action_var = tf.cast(state_info_vars["prev_action"], tf.float32)
            all_input_var = tf.concat(2, [obs_var, prev_action_var])
        else:
            all_input_var = obs_var

        if self.policy.feature_network is None:
            vals = L.get_output(
                self.value_network.output_layer,
                {self.policy.l_input: all_input_var},
                **kwargs
            )
        else:
            flat_input_var = tf.reshape(all_input_var, (-1, self.policy.input_dim))
            vals = L.get_output(
                self.value_network.output_layer,
                {self.policy.l_input: all_input_var, self.policy.feature_network.input_layer: flat_input_var},
                **kwargs
            )
        vals = tf.reshape(vals, tf.pack([tf.shape(obs_var)[0], tf.shape(obs_var)[1]]))
        return vals

    def predict_sym_reuse(self, state_output):
        vals = L.get_output(
            self.value_network.output_layer,
            {self.l_rnn: state_output},
        )
        vals = tf.reshape(vals, tf.pack([tf.shape(state_output)[0], tf.shape(state_output)[1]]))
        return vals

    def predict(self, path):
        obs = path["observations"]
        agent_infos = path["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        vfs = self.f_predict([obs], *[[x] for x in state_info_list])[0]
        return vfs

    def fit(self, paths):
        pass

    def log_diagnostics(self, paths):
        pass


class PPOSGD(BatchPolopt):
    def __init__(
            self,
            step_size=0.01,
            n_steps=20,
            n_epochs=10,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            entropy_bonus_coeff=0.,
            vf_loss_coeff=1.,
            **kwargs
    ):
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.vf_loss_coeff = vf_loss_coeff
        super().__init__(**kwargs)

    def init_opt(self):
        assert self.policy.recurrent

        vf = self.baseline

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=2,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=2,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=2,
            dtype=tf.float32,
        )
        returns_var = tensor_utils.new_tensor(
            'return',
            ndim=2,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None, None) + shape, name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None, None) + shape, name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        valid_var = tf.placeholder(tf.float32, shape=(None, None), name="valid")

        rnn_network = self.policy.prob_network

        state_var = tf.placeholder(tf.float32, (None, rnn_network.state_dim), "state")

        kl_penalty_var = tf.placeholder(tf.float32, shape=(), name="kl_penalty")

        recurrent_layer = rnn_network.recurrent_layer
        recurrent_state_output = dict()

        minibatch_dist_info_vars = self.policy.dist_info_sym(
            obs_var, state_info_vars,
            recurrent_state={recurrent_layer: state_var},
            recurrent_state_output=recurrent_state_output,
        )

        state_output = recurrent_state_output[rnn_network.recurrent_layer]
        final_state = tf.reverse(state_output, [False, True, False])[:, 0, :]

        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, minibatch_dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, minibatch_dist_info_vars)
        ent = tf.reduce_sum(dist.entropy_sym(minibatch_dist_info_vars) * valid_var) / tf.reduce_sum(valid_var)
        mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
        surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)

        predicted_returns_var = vf.predict_sym_reuse(
            state_output
        )

        mean_returns = tf.reduce_sum(returns_var * valid_var) / tf.reduce_sum(valid_var)
        vf_std = tf.sqrt(tf.reduce_sum(tf.square(returns_var - mean_returns) * valid_var) / tf.reduce_sum(valid_var))

        vf_rescaled_diff = (returns_var - predicted_returns_var) / (vf_std + 1e-5)

        vf_loss = tf.reduce_sum(tf.square(vf_rescaled_diff) * valid_var) / tf.reduce_sum(valid_var)

        surr_pen_loss = surr_loss + kl_penalty_var * tf.maximum(0., mean_kl - self.step_size) - \
                        self.entropy_bonus_coeff * ent + self.vf_loss_coeff * vf_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        params = list(set(self.policy.get_params(trainable=True) + vf.get_params(trainable=True)))
        train_op = optimizer.minimize(surr_pen_loss, var_list=params)

        self.f_train = tensor_utils.compile_function(
            inputs=[obs_var, action_var, advantage_var, returns_var] + state_info_vars_list + old_dist_info_vars_list + \
                   [valid_var, state_var, kl_penalty_var],
            outputs=[train_op, surr_loss, mean_kl, vf_loss, final_state],
        )
        self.f_loss_kl = tensor_utils.compile_function(
            inputs=[obs_var, action_var, advantage_var, returns_var] + state_info_vars_list + old_dist_info_vars_list + \
                   [valid_var, state_var],
            outputs=[surr_loss, mean_kl, vf_loss],
        )
        self.joint_parameterized = JointParameterized([self.policy, vf])

    def optimize_policy(self, itr, samples_data):

        observations = samples_data["observations"]
        actions = samples_data["actions"]
        advantages = samples_data["advantages"]
        valids = samples_data["valids"]
        returns = samples_data["returns"]

        # self.baseline.renormalize(returns, valids)

        # renormalize vf
        # import ipdb; ipdb.set_trace()

        # Perform truncated backprop
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        all_inputs = [observations, actions, advantages, returns] + state_info_list + dist_info_list + [valids]
        # all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        # if self.policy.recurrent:
        #     all_input_values += (samples_data["valids"],)

        N, T, _ = observations.shape
        if self.n_steps is None:
            n_steps = T
        else:
            n_steps = self.n_steps

        init_states = np.tile(
            self.policy.prob_network.state_init_param.eval().reshape((1, -1)),
            (N, 1)
        )

        surr_loss_before, kl_before, vf_loss_before = self.f_loss_kl(*(all_inputs + [init_states]))

        kl_penalty = 1.

        best_loss = None
        best_params = None

        for epoch_id in range(self.n_epochs):
            states = init_states
            surr_losses = []
            mean_kls = []
            vf_losses = []
            for t in range(0, T, n_steps):
                sliced_inputs = [x[:, t:t + n_steps] for x in all_inputs]
                _, surr_loss, mean_kl, vf_loss, states = self.f_train(*(sliced_inputs + [states, kl_penalty]))
                surr_losses.append(surr_loss)
                mean_kls.append(mean_kl)
                vf_losses.append(vf_loss)
            mean_kl = np.mean(mean_kls)
            surr_loss = np.mean(surr_losses)
            vf_loss = np.mean(vf_losses)
            logger.log("Loss: %f; Mean KL: %f; Vf loss: %f; KL penalty: %f" % (surr_loss, mean_kl, vf_loss, kl_penalty))
            if mean_kl > self.step_size:
                kl_penalty *= self.increase_penalty_factor
            else:
                kl_penalty *= self.decrease_penalty_factor
            if mean_kl <= self.step_size:
                if best_loss is None or surr_loss + self.vf_loss_coeff * vf_loss < best_loss:
                    best_loss = surr_loss + self.vf_loss_coeff * vf_loss
                    best_params = self.joint_parameterized.get_param_values()

        if best_params is not None:
            self.joint_parameterized.set_param_values(best_params)

        surr_loss_after, kl_after, vf_loss_after = self.f_loss_kl(*(all_inputs + [init_states]))

        # perform minibatch gradient descent on the surrogate loss, while monitoring the KL divergence

        logger.record_tabular('SurrLossBefore', surr_loss_before)
        logger.record_tabular('SurrLossAfter', surr_loss_after)
        logger.record_tabular('VfLossBefore', vf_loss_before)
        logger.record_tabular('VfLossAfter', vf_loss_after)
        logger.record_tabular('MeanKLBefore', kl_before)
        logger.record_tabular('MeanKL', kl_after)
        logger.record_tabular('dSurrLoss', surr_loss_before - surr_loss_after)
        logger.record_tabular('dVfLoss', vf_loss_before - vf_loss_after)
        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
