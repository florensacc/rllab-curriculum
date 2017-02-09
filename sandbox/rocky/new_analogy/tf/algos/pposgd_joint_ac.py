from collections import OrderedDict

from rllab.misc import logger
from sandbox.rocky.neural_learner.optimizers.sgd_optimizer import SGDOptimizer
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
            use_line_search=True,
            max_backtracks=10,
            backtrack_ratio=0.5,
            optimizer=None,
            step_size=0.01,
            min_n_epochs=2,
            adaptive_learning_rate=False,
            max_learning_rate=1e-3,
            min_learning_rate=1e-5,
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
        self.use_line_search = use_line_search
        self.max_backtracks = max_backtracks
        self.backtrack_ratio = backtrack_ratio
        self.step_size = step_size
        self.min_n_epochs = min_n_epochs
        self.adaptive_learning_rate = adaptive_learning_rate
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        policy = kwargs['policy']
        if optimizer is None:
            if policy.recurrent:
                optimizer = TBPTTOptimizer()
            else:
                optimizer = SGDOptimizer()
        self.optimizer = optimizer
        super().__init__(**kwargs)

    def init_opt(self):

        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        returns_var = tensor_utils.new_tensor(
            'returns',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None,) * (1 + is_recurrent) + shape, name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None,) * (1 + is_recurrent) + shape, name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        kl_penalty_var = tf.Variable(
            initial_value=self.initial_kl_penalty,
            dtype=tf.float32,
            name="kl_penalty"
        )

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=(None, None), name="valid")

            if hasattr(self.policy, "head_network"):
                rnn_network = self.policy.head_network
                state_dim = rnn_network.state_dim
                recurrent_layer = rnn_network.recurrent_layer
                state_init_param = rnn_network.state_init_param
            else:
                state_dim = self.policy.l_rnn.state_dim
                recurrent_layer = self.policy.l_rnn
                state_init_param = tf.reshape(self.policy.l_rnn.cell.zero_state(1, dtype=tf.float32), (-1,))

            state_var = tf.placeholder(tf.float32, (None, state_dim), "state")

            recurrent_state_output = dict()

            minibatch_dist_info_vars = self.policy.dist_info_sym(
                obs_var, state_info_vars,
                recurrent_state={recurrent_layer: state_var},
                recurrent_state_output=recurrent_state_output,
            )

            state_output = recurrent_state_output[recurrent_layer]

            if hasattr(self.policy, "head_network"):
                final_state = tf.reverse(state_output, [False, True, False])[:, 0, :]
            else:
                final_state = state_output

            lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, minibatch_dist_info_vars)
            kl = dist.kl_sym(old_dist_info_vars, minibatch_dist_info_vars)
            ent = tf.reduce_sum(dist.entropy_sym(minibatch_dist_info_vars) * valid_var) / (tf.reduce_sum(valid_var) +
                                                                                           1e-8)
            mean_kl = tf.reduce_sum(kl * valid_var) / (tf.reduce_sum(valid_var) + 1e-8)

            clipped_lr = tf.clip_by_value(lr, 1. - self.clip_lr, 1. + self.clip_lr)

            surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / (tf.reduce_sum(valid_var) + 1e-8)
            clipped_surr_loss = - tf.reduce_sum(
                tf.minimum(lr * advantage_var, clipped_lr * advantage_var) * valid_var
            ) / (tf.reduce_sum(valid_var) + 1e-8)

            clipped_surr_pen_loss = clipped_surr_loss - self.entropy_bonus_coeff * ent
            if self.use_kl_penalty:
                clipped_surr_pen_loss += kl_penalty_var * tf.maximum(0., mean_kl - self.step_size)

            policy_loss = clipped_surr_pen_loss

            vf_predicted = minibatch_dist_info_vars["vf"][:, :, 0]
            vf_actual = returns_var

            vf_loss = tf.reduce_sum(
                tf.square((vf_predicted - vf_actual) / (self.policy.return_std_var + 1e-8)) * valid_var
            ) / (tf.reduce_sum(valid_var) + 1e-8)

            total_loss = policy_loss + 0.5 * vf_loss

            self.optimizer.update_opt(
                loss=total_loss,
                target=self.policy,
                inputs=[obs_var, action_var, advantage_var, returns_var] + state_info_vars_list + \
                       old_dist_info_vars_list + [valid_var],
                rnn_init_state=state_init_param,
                rnn_state_input=state_var,
                rnn_final_state=final_state,
                diagnostic_vars=OrderedDict([
                    ("UnclippedSurrLoss", surr_loss),
                    ("MeanKL", mean_kl),
                    # ("mean", minibatch_dist_info_vars["mean"]),
                    # ("log_std", minibatch_dist_info_vars["log_std"]),
                    # ("lr", lr),
                    # ("valid", valid_var),
                    # ("clipped_surr_pen_loss", clipped_surr_pen_loss),
                    # ("vf_loss", vf_loss),
                    # ("vf_predicted", vf_predicted),
                    # ("vf_actual", vf_actual),
                    # ("return_std", self.policy.return_std_var),
                ])
            )
        else:
            # raise NotImplementedError
            dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)

            lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
            ent = tf.reduce_mean(dist.entropy_sym(dist_info_vars))
            mean_kl = tf.reduce_mean(kl)

            clipped_lr = tf.clip_by_value(lr, 1. - self.clip_lr, 1. + self.clip_lr)

            surr_loss = - tf.reduce_mean(lr * advantage_var)
            clipped_surr_loss = - tf.reduce_mean(
                tf.minimum(lr * advantage_var, clipped_lr * advantage_var)
            )

            clipped_surr_pen_loss = clipped_surr_loss - self.entropy_bonus_coeff * ent
            if self.use_kl_penalty:
                clipped_surr_pen_loss += kl_penalty_var * tf.maximum(0., mean_kl - self.step_size)

            policy_loss = clipped_surr_pen_loss

            vf_predicted = dist_info_vars["vf"][:, 0]
            vf_actual = returns_var

            vf_loss = tf.reduce_mean(
                tf.square((vf_predicted - vf_actual) / (self.policy.return_std_var + 1e-8))
            )

            total_loss = policy_loss + 0.5 * vf_loss

            self.optimizer.update_opt(
                loss=total_loss,
                target=self.policy,
                inputs=[obs_var, action_var, advantage_var, returns_var] + state_info_vars_list + old_dist_info_vars_list,
                diagnostic_vars=OrderedDict([
                    ("UnclippedSurrLoss", surr_loss),
                    ("MeanKL", mean_kl),
                ])
            )

        self.kl_penalty_var = kl_penalty_var
        self.f_increase_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                kl_penalty_var,
                tf.minimum(kl_penalty_var * self.increase_penalty_factor, self.max_penalty)
            )
        )
        self.f_decrease_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                kl_penalty_var,
                tf.maximum(kl_penalty_var * self.decrease_penalty_factor, self.min_penalty)
            )
        )
        self.f_reset_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                kl_penalty_var,
                self.initial_kl_penalty
            )
        )

    def sliced_loss_kl(self, inputs):
        loss, diags = self.optimizer.loss_diagnostics(inputs=inputs)
        return diags["UnclippedSurrLoss"], diags["MeanKL"]

    def optimize_policy(self, itr, samples_data):
        logger.log("Policy param norm: %f" % np.linalg.norm(self.policy.get_param_values()))
        logger.log("Start optimizing..")
        observations = samples_data["observations"]
        actions = samples_data["actions"]
        advantages = samples_data["advantages"]
        returns = samples_data["returns"]

        # Perform truncated backprop
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        all_inputs = [observations, actions, advantages, returns] + state_info_list + dist_info_list

        if self.policy.recurrent:
            valids = samples_data["valids"]
            all_inputs.append(valids)

        if self.log_loss_kl_before:
            logger.log("Computing loss / KL before training")
            surr_loss_before, kl_before = self.sliced_loss_kl(all_inputs)
            logger.log("Computed")

        prev_params = self.policy.get_param_values(trainable=True)

        epoch_surr_losses = []
        epoch_mean_kls = []

        best_loss = None
        best_kl = None
        best_params = None

        self.f_reset_penalty()

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

        if self.log_loss_kl_after or self.use_line_search or self.adaptive_learning_rate:
            logger.log("Computing loss / KL after training")
            surr_loss_after, kl_after = self.sliced_loss_kl(all_inputs)
            logger.log("Computed")

            if self.adaptive_learning_rate:
                if kl_after > self.step_size:
                    self.optimizer.learning_rate = max(self.min_learning_rate, self.optimizer.learning_rate * 0.5)
                else:
                    self.optimizer.learning_rate = min(self.max_learning_rate, self.optimizer.learning_rate * 2)

            if self.use_line_search and kl_after > self.step_size:
                logger.log("Performing line search to make sure KL is within range")
                n_trials = 0
                step_size = 1.
                now_params = self.policy.get_param_values(trainable=True)
                while kl_after > self.step_size and n_trials < self.max_backtracks:
                    step_size *= self.backtrack_ratio
                    self.policy.set_param_values(
                        (1 - step_size) * prev_params + step_size * now_params,
                        trainable=True
                    )
                    n_trials += 1
                    surr_loss_after, kl_after = self.sliced_loss_kl(all_inputs)
                    logger.log("After shrinking step, loss = %f, Mean KL = %f" % (surr_loss_after, kl_after))

        # perform minibatch gradient descent on the surrogate loss, while monitoring the KL divergence

        now_params = self.policy.get_param_values(trainable=True)

        logger.record_tabular('dPolicyParamNorm', np.linalg.norm(now_params - prev_params))
        logger.record_tabular('PolicyParamNorm', np.linalg.norm(now_params))
        logger.record_tabular('LearningRate', self.optimizer.learning_rate)

        if self.log_loss_kl_before:
            logger.record_tabular('SurrLossBefore', surr_loss_before)
            logger.record_tabular('MeanKLBefore', kl_before)
        else:
            # Log approximately
            logger.record_tabular('FirstEpoch.SurrLoss', epoch_surr_losses[0])
            logger.record_tabular('FirstEpoch.MeanKL', epoch_mean_kls[0])
        if self.log_loss_kl_after or self.use_line_search:
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
