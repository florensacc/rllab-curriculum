from collections import OrderedDict

from rllab.misc import logger
from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils
import numpy as np
import sys


class PPOSGD(BatchPolopt):
    def __init__(
            self,
            clip_lr=0.3,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            min_penalty=1e-3,
            max_penalty=1e6,
            entropy_bonus_coeff=0.,
            gradient_clipping=40.,
            log_loss_kl_before=True,
            log_loss_kl_after=True,
            use_kl_penalty=False,
            initial_kl_penalty=1.,
            optimizer=None,
            step_size=0.01,
            min_n_epochs=2,
            **kwargs
    ):
        self.clip_lr = clip_lr
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.gradient_clipping = gradient_clipping
        self.log_loss_kl_before = log_loss_kl_before
        self.log_loss_kl_after = log_loss_kl_after
        self.use_kl_penalty = use_kl_penalty
        self.initial_kl_penalty = initial_kl_penalty
        self.step_size = step_size
        self.min_n_epochs = min_n_epochs
        if optimizer is None:
            optimizer = TBPTTOptimizer()
        self.optimizer = optimizer

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

        state_output = recurrent_state_output[rnn_network.recurrent_layer]
        final_state = tf.reverse(state_output, [False, True, False])[:, 0, :]

        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, minibatch_dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, minibatch_dist_info_vars)
        ent = tf.reduce_sum(dist.entropy_sym(minibatch_dist_info_vars) * valid_var) / tf.reduce_sum(valid_var)
        mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
        kl_penalty_var = tf.Variable(
            initial_value=self.initial_kl_penalty,
            dtype=tf.float32,
            name="kl_penalty"
        )

        clipped_lr = tf.clip_by_value(lr, 1. - self.clip_lr, 1. + self.clip_lr)

        surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        clipped_surr_loss = - tf.reduce_sum(
            tf.minimum(lr * advantage_var, clipped_lr * advantage_var) * valid_var
        ) / tf.reduce_sum(valid_var)

        clipped_surr_pen_loss = clipped_surr_loss - self.entropy_bonus_coeff * ent
        if self.use_kl_penalty:
            clipped_surr_pen_loss += kl_penalty_var * tf.maximum(0., mean_kl - self.step_size)

        self.kl_penalty_var = kl_penalty_var

        self.optimizer.update_opt(
            loss=clipped_surr_pen_loss,
            target=self.policy,
            inputs=[obs_var, action_var, advantage_var] + state_info_vars_list + old_dist_info_vars_list + [valid_var],
            rnn_init_state=rnn_network.state_init_param,
            rnn_state_input=state_var,
            rnn_final_state=final_state,
            diagnostic_vars=OrderedDict([
                ("UnclippedSurrLoss", surr_loss),
                ("MeanKL", mean_kl),
            ])
        )

        self.f_increase_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                self.kl_penalty_var,
                tf.minimum(self.kl_penalty_var * self.increase_penalty_factor, self.max_penalty)
            )
        )
        self.f_decrease_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                self.kl_penalty_var,
                tf.maximum(self.kl_penalty_var * self.decrease_penalty_factor, self.min_penalty)
            )
        )

    def sliced_loss_kl(self, inputs):
        loss, diags = self.optimizer.loss_diagostics(inputs)
        return diags["UnclippedSurrLoss"], diags["MeanKL"]

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

        if self.log_loss_kl_before:
            logger.log("Computing loss / KL before training")
            surr_loss_before, kl_before = self.sliced_loss_kl(all_inputs)
            logger.log("Computed")

        epoch_surr_losses = []
        epoch_mean_kls = []

        best_loss = None
        best_kl = None
        best_params = None

        def itr_callback(itr, loss, diagnostics, *args, **kwargs):
            nonlocal best_loss
            nonlocal best_params
            nonlocal best_kl
            surr_loss = diagnostics["UnclippedSurrLoss"]
            mean_kl = diagnostics["MeanKL"]
            epoch_surr_losses.append(surr_loss)
            epoch_mean_kls.append(mean_kl)
            if mean_kl <= self.step_size:
                if best_loss is None or surr_loss < best_loss:
                    best_loss = surr_loss
                    best_kl = mean_kl
                    best_params = self.policy.get_param_values()
            if mean_kl <= self.step_size and itr + 1 >= self.min_n_epochs:
                penalty = self.f_decrease_penalty()
                logger.log("Epoch %d; Loss %f; Mean KL: %f; decreasing penalty to %f and finish opt since KL and "
                           "minimum #epochs reached" % (itr, surr_loss, mean_kl, penalty))
                # early termination
                return False
            if self.use_kl_penalty:
                if mean_kl > self.step_size:
                    # constraint violated. increase penalty
                    penalty = self.f_increase_penalty()
                    logger.log("Epoch %d; Loss %f; Mean KL: %f; increasing penalty to %f" % (
                        itr, surr_loss, mean_kl, penalty))
                else:
                    penalty = self.f_decrease_penalty()
                    logger.log("Epoch %d; Loss %f; Mean KL: %f; decreasing penalty to %f" % (
                        itr, surr_loss, mean_kl, penalty))
            elif itr + 1 >= self.min_n_epochs:
                # if do not use kl penalty, only execute for the minimum number of epochs
                return False
            return True

        self.optimizer.optimize(all_inputs, callback=itr_callback)

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
            if best_loss is None:
                logger.record_tabular('LastEpoch.SurrLoss', epoch_surr_losses[-1])
                logger.record_tabular('LastEpoch.MeanKL', epoch_mean_kls[-1])
            else:
                logger.record_tabular('LastEpoch.SurrLoss', best_loss)
                logger.record_tabular('LastEpoch.MeanKL', best_kl)
        if self.log_loss_kl_before and self.log_loss_kl_after:
            logger.record_tabular('dSurrLoss', surr_loss_before - surr_loss_after)

        if np.isnan(epoch_surr_losses[-1]):
            logger.log("NaN detected! Terminating")
            sys.exit()

        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
