from rllab.core.serializable import Serializable
from rllab.misc import logger
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
import tensorflow as tf

from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.misc import tensor_utils
import numpy as np


class PPOSGD(BatchPolopt):
    def __init__(
            self,
            clip_lr=0.3,
            minibatch_size=256,
            n_steps=20,
            n_epochs=10,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            entropy_bonus_coeff=0.,
            gradient_clipping=40.,
            log_loss_kl_before=True,
            log_loss_kl_after=True,
            **kwargs
    ):
        self.clip_lr = clip_lr
        self.minibatch_size = minibatch_size
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.gradient_clipping = gradient_clipping
        self.log_loss_kl_before = log_loss_kl_before
        self.log_loss_kl_after = log_loss_kl_after
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

        recurrent_layer = rnn_network.recurrent_layer
        recurrent_state_output = dict()

        minibatch_dist_info_vars = self.policy.dist_info_sym(
            obs_var, state_info_vars,
            recurrent_state={recurrent_layer: state_var},
            recurrent_state_output=recurrent_state_output,
        )
        minibatch_dist_info_vars_list = [minibatch_dist_info_vars[k] for k in dist.dist_info_keys]

        state_output = recurrent_state_output[rnn_network.recurrent_layer]
        final_state = tf.reverse(state_output, [False, True, False])[:, 0, :]

        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, minibatch_dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, minibatch_dist_info_vars)
        ent = tf.reduce_sum(dist.entropy_sym(minibatch_dist_info_vars) * valid_var) / tf.reduce_sum(valid_var)
        mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)

        clipped_lr = tf.clip_by_value(lr, 1. - self.clip_lr, 1. + self.clip_lr)

        surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        clipped_surr_loss = - tf.reduce_sum(
            tf.minimum(lr * advantage_var, clipped_lr * advantage_var) * valid_var
        ) / tf.reduce_sum(valid_var)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        params = self.policy.get_params(trainable=True)
        gvs = optimizer.compute_gradients(clipped_surr_loss, var_list=params)
        if self.gradient_clipping is not None:
            capped_gvs = [(tf.clip_by_value(grad, -self.gradient_clipping, self.gradient_clipping), var) for grad, var
                          in gvs]
        else:
            capped_gvs = gvs
        train_op = optimizer.apply_gradients(capped_gvs)

        self.f_train = tensor_utils.compile_function(
            inputs=[obs_var, action_var, advantage_var] + state_info_vars_list + old_dist_info_vars_list + \
                   [valid_var, state_var],
            outputs=[train_op, surr_loss, mean_kl, final_state],
        )
        self.f_loss_kl = tensor_utils.compile_function(
            inputs=[obs_var, action_var, advantage_var] + state_info_vars_list + old_dist_info_vars_list + \
                   [valid_var, state_var],
            outputs=[surr_loss, mean_kl, final_state],
        )
        self.f_debug = tensor_utils.compile_function(
            inputs=[obs_var, action_var, advantage_var] + state_info_vars_list + old_dist_info_vars_list + \
                   [valid_var, state_var],
            outputs=[surr_loss, mean_kl, final_state, lr, kl, minibatch_dist_info_vars_list],
        )

    def sliced_loss_kl(self, inputs):
        N, T, _ = inputs[0].shape
        if self.n_steps is None:
            n_steps = T
        else:
            n_steps = self.n_steps
        if self.minibatch_size is None:
            minibatch_size = N
        else:
            minibatch_size = self.minibatch_size

        surr_losses = []
        mean_kls = []

        for batch_idx in range(0, N, minibatch_size):
            batch_sliced_inputs = [x[batch_idx:batch_idx + self.minibatch_size] for x in inputs]
            states = np.tile(
                self.policy.prob_network.state_init_param.eval().reshape((1, -1)),
                (batch_sliced_inputs[0].shape[0], 1)
            )
            for t in range(0, T, n_steps):
                time_sliced_inputs = [x[:, t:t + n_steps] for x in batch_sliced_inputs]
                surr_loss, mean_kl, states = self.f_loss_kl(*(time_sliced_inputs + [states]))
                surr_losses.append(surr_loss)
                mean_kls.append(mean_kl)
        return np.mean(surr_losses), np.mean(mean_kls)

    def optimize_policy(self, itr, samples_data):
        logger.log("Start optimizing..")
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
        if self.minibatch_size is None:
            minibatch_size = N
        else:
            minibatch_size = self.minibatch_size

        if self.log_loss_kl_before:
            logger.log("Computing loss / KL before training")
            surr_loss_before, kl_before = self.sliced_loss_kl(all_inputs)
            logger.log("Computed")

        best_loss = None
        best_params = None

        logger.log("Start training...")

        epoch_surr_losses = []
        epoch_mean_kls = []

        for epoch_id in range(self.n_epochs):
            logger.log("Epoch %d" % epoch_id)
            surr_losses = []
            mean_kls = []
            for batch_idx in range(0, N, minibatch_size):
                batch_sliced_inputs = [x[batch_idx:batch_idx + self.minibatch_size] for x in all_inputs]
                states = np.tile(
                    self.policy.prob_network.state_init_param.eval().reshape((1, -1)),
                    (batch_sliced_inputs[0].shape[0], 1)
                )
                for t in range(0, T, n_steps):
                    time_sliced_inputs = [x[:, t:t + n_steps] for x in batch_sliced_inputs]
                    # The last input is the valid mask. Only bother computing if at least one entry is valid
                    if np.any(np.nonzero(time_sliced_inputs[-1])):
                        old_states = states
                        _, surr_loss, mean_kl, states = self.f_train(*(time_sliced_inputs + [states]))
                        surr_losses.append(surr_loss)
                        mean_kls.append(mean_kl)
                        if np.isnan(surr_loss) or np.isnan(mean_kl):
                            debug_vals = self.f_debug(*(time_sliced_inputs + [old_states]))
                            # TODO: replace this !!
                            import ipdb; ipdb.set_trace()
                    else:
                        break
            mean_kl = np.mean(mean_kls)
            surr_loss = np.mean(surr_losses)
            logger.log("Loss: %f; Mean KL: %f" % (surr_loss, mean_kl))

            epoch_surr_losses.append(surr_loss)
            epoch_mean_kls.append(mean_kl)

            if best_loss is None or surr_loss < best_loss:
                best_loss = surr_loss
                best_params = self.policy.get_param_values()

        if best_params is not None:
            self.policy.set_param_values(best_params)

        if self.log_loss_kl_after:
            logger.log("Computing loss / KL after training")
            surr_loss_after, kl_after = self.sliced_loss_kl(all_inputs)
            logger.log("Computed")

        # perform minibatch gradient descent on the surrogate loss, while monitoring the KL divergence

        if self.log_loss_kl_before:
            logger.record_tabular('SurrLossBefore', surr_loss_before)
            logger.record_tabular('MeanKLBefore', kl_before)
        else:
            # Log approximately
            logger.record_tabular('FirstEpoch.SurrLoss', epoch_surr_losses[0])
            logger.record_tabular('FirstEpoch.MeanKL', epoch_mean_kls[0])
        if self.log_loss_kl_after:
            logger.record_tabular('SurrLossAfter', surr_loss_after)
            logger.record_tabular('MeanKL', kl_after)
        else:
            logger.record_tabular('LastEpoch.SurrLoss', epoch_surr_losses[-1])
            logger.record_tabular('LastEpoch.MeanKL', epoch_mean_kls[-1])
        if self.log_loss_kl_before and self.log_loss_kl_after:
            logger.record_tabular('dSurrLoss', surr_loss_before - surr_loss_after)

        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
