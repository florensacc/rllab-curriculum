from rllab.core.serializable import Serializable
from rllab.misc import logger
import tensorflow as tf

from sandbox.rocky.neural_learner.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.misc import tensor_utils
import numpy as np


class PPOSGD(BatchPolopt):
    def __init__(
            self,
            step_size=0.01,
            n_steps=20,
            n_epochs=10,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            entropy_bonus_coeff=0.,
            **kwargs
    ):
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.entropy_bonus_coeff = entropy_bonus_coeff
        super().__init__(**kwargs)

    def init_opt(self):
        assert self.policy.recurrent

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

        surr_pen_loss = surr_loss + kl_penalty_var * tf.maximum(0., mean_kl - self.step_size) - \
                        self.entropy_bonus_coeff * ent

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        params = self.policy.get_params(trainable=True)
        train_op = optimizer.minimize(surr_pen_loss, var_list=params)

        self.f_train = tensor_utils.compile_function(
            inputs=[obs_var, action_var, advantage_var] + state_info_vars_list + old_dist_info_vars_list + \
                   [valid_var, state_var, kl_penalty_var],
            outputs=[train_op, surr_loss, mean_kl, final_state],
        )
        self.f_loss_kl = tensor_utils.compile_function(
            inputs=[obs_var, action_var, advantage_var] + state_info_vars_list + old_dist_info_vars_list + \
                   [valid_var, state_var],
            outputs=[surr_loss, mean_kl],
        )

    def optimize_policy(self, itr, samples_data):

        observations = samples_data["observations"]
        actions = samples_data["actions"]
        advantages = samples_data["advantages"]
        valids = samples_data["valids"]

        # Perform truncated backprop
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        all_inputs = [observations, actions, advantages] + state_info_list + dist_info_list + [valids]
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

        surr_loss_before, kl_before = self.f_loss_kl(*(all_inputs + [init_states]))

        kl_penalty = 1.

        best_loss = None
        best_params = None

        for epoch_id in range(self.n_epochs):
            states = init_states
            surr_losses = []
            mean_kls = []
            for t in range(0, T, n_steps):
                sliced_inputs = [x[:, t:t + n_steps] for x in all_inputs]
                _, surr_loss, mean_kl, states = self.f_train(*(sliced_inputs + [states, kl_penalty]))
                surr_losses.append(surr_loss)
                mean_kls.append(mean_kl)
            mean_kl = np.mean(mean_kls)
            surr_loss = np.mean(surr_losses)
            logger.log("Loss: %f; Mean KL: %f; KL penalty: %f" % (surr_loss, mean_kl, kl_penalty))
            if mean_kl > self.step_size:
                kl_penalty *= self.increase_penalty_factor
            else:
                kl_penalty *= self.decrease_penalty_factor
            if mean_kl <= self.step_size:
                if best_loss is None or surr_loss < best_loss:
                    best_loss = surr_loss
                    best_params = self.policy.get_param_values()

        if best_params is not None:
            self.policy.set_param_values(best_params)

        surr_loss_after, kl_after = self.f_loss_kl(*(all_inputs + [init_states]))

        # perform minibatch gradient descent on the surrogate loss, while monitoring the KL divergence

        logger.record_tabular('SurrLossBefore', surr_loss_before)
        logger.record_tabular('SurrLossAfter', surr_loss_after)
        logger.record_tabular('MeanKLBefore', kl_before)
        logger.record_tabular('MeanKL', kl_after)
        logger.record_tabular('dSurrLoss', surr_loss_before - surr_loss_after)
        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
