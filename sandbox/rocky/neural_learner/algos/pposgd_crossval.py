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


class PPOSGD(BatchPolopt):
    def __init__(
            self,
            step_size=0.01,
            n_steps=20,
            n_epochs=10,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            entropy_bonus_coeff=0.,
            train_ratio=0.7,
            **kwargs
    ):
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.train_ratio = train_ratio
        self.unused_paths = []
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

        params = list(set(self.policy.get_params(trainable=True)))
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

    def obtain_samples(self, itr):
        paths = super().obtain_samples(itr)
        return self.unused_paths + paths

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

        N, T, _ = observations.shape
        if self.n_steps is None:
            n_steps = T
        else:
            n_steps = self.n_steps


        train_N = int(N * self.train_ratio)
        val_N = N - train_N

        train_init_states = np.tile(
            self.policy.prob_network.state_init_param.eval().reshape((1, -1)),
            (train_N, 1)
        )
        val_init_states = np.tile(
            self.policy.prob_network.state_init_param.eval().reshape((1, -1)),
            (val_N, 1)
        )

        train_inputs = [x[:train_N] for x in all_inputs]
        val_inputs = [x[train_N:] for x in all_inputs]

        train_surr_loss_before, train_kl_before = self.f_loss_kl(*(train_inputs + [train_init_states]))
        val_surr_loss_before, val_kl_before = self.f_loss_kl(*(val_inputs + [val_init_states]))

        kl_penalty = 1.

        best_params = self.policy.get_param_values()#None
        best_val_loss = val_surr_loss_before
        updated = False

        logger.log("Val surr loss: %f; Val Mean KL: %f" % (val_surr_loss_before, val_kl_before))

        for epoch_id in range(self.n_epochs):
            states = train_init_states
            surr_losses = []
            mean_kls = []
            for t in range(0, T, n_steps):
                sliced_inputs = [x[:, t:t + n_steps] for x in train_inputs]
                _, surr_loss, mean_kl, states = self.f_train(*(sliced_inputs + [states, kl_penalty]))
                surr_losses.append(surr_loss)
                mean_kls.append(mean_kl)
            mean_kl = np.mean(mean_kls)
            surr_loss = np.mean(surr_losses)

            val_surr_loss, val_mean_kl = self.f_loss_kl(*(val_inputs + [val_init_states]))

            logger.log("Train Loss: %f; Val Loss: %f; Train Mean KL: %f; KL penalty: %f" % (surr_loss, val_surr_loss,
                                                                                            mean_kl, kl_penalty))
            if val_surr_loss > best_val_loss:
                kl_penalty *= self.increase_penalty_factor
            else:
                kl_penalty *= self.decrease_penalty_factor
            # Evaluate the validation loss

            # logger.log("Val surr loss: %f; Val Mean KL: %f" % (val_surr_loss, val_mean_kl))

            if val_surr_loss < best_val_loss:
                best_val_loss = val_surr_loss
                best_params = self.policy.get_param_values()
                updated = True
            # else:

        # If the policy is not updated, we can reuse the samples collected from last time
        self.policy.set_param_values(best_params)
        if updated:
            self.unused_paths = []
        else:
            # reuse paths
            self.unused_paths += samples_data["paths"]
                # break


            # if mean_kl <= self.step_size:
            #     if best_loss is None or surr_loss < best_loss:
            #         best_loss = surr_loss
            #         best_params = self.policy.get_param_values()

        # if best_params is not None:
        #     self.policy.set_param_values(best_params)

        train_surr_loss_after, train_kl_after = self.f_loss_kl(*(train_inputs + [train_init_states]))
        val_surr_loss_after, val_kl_after = self.f_loss_kl(*(val_inputs + [val_init_states]))

        # perform minibatch gradient descent on the surrogate loss, while monitoring the KL divergence

        logger.record_tabular('TrainSurrLossBefore', train_surr_loss_before)
        logger.record_tabular('TrainSurrLossAfter', train_surr_loss_after)
        logger.record_tabular('TrainMeanKLBefore', train_kl_before)
        logger.record_tabular('TrainMeanKL', train_kl_after)
        logger.record_tabular('TrainDSurrLoss', train_surr_loss_before - train_surr_loss_after)
        logger.record_tabular('ValSurrLossBefore', val_surr_loss_before)
        logger.record_tabular('ValSurrLossAfter', val_surr_loss_after)
        logger.record_tabular('ValMeanKLBefore', val_kl_before)
        logger.record_tabular('ValMeanKL', val_kl_after)
        logger.record_tabular('ValDSurrLoss', val_surr_loss_before - val_surr_loss_after)
        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
