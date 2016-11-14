from collections import OrderedDict
import theano
import theano.tensor as TT
import numpy as np
import sys

from rllab.misc import logger, console
from rllab.algos.batch_polopt import BatchPolopt
from rllab.misc import ext
from sandbox.haoran.parallel_trpo.sgd_optimizer_theano import SGDOptimizer


class PPOSGD(BatchPolopt):
    """
    Theano version of Rocky's implementation in neural_learner/algos/
    key coeffs: use_kl_penalty, step_size, min_n_epochs, clip_lr, optimizer.gradient_clipping
    """
    def __init__(
            self,
            clip_lr=0.3,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            min_penalty=1e-3,
            max_penalty=1e6,
            entropy_bonus_coeff=0.,
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
            log_prefix="pposgd: ",
            **kwargs
    ):
        self.clip_lr = clip_lr
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.log_loss_kl_before = log_loss_kl_before
        self.log_loss_kl_after = log_loss_kl_after
        self.use_kl_penalty = use_kl_penalty
        self.initial_kl_penalty = np.float32(initial_kl_penalty)
        self.use_line_search = use_line_search
        self.max_backtracks = max_backtracks
        self.backtrack_ratio = backtrack_ratio
        self.step_size = step_size
        self.min_n_epochs = min_n_epochs
        self.optimizer = optimizer
        self.log_prefix = log_prefix
        super().__init__(**kwargs)

    def log(self, message, color=None):
        if color is not None:
            message = console.colorize(message, color)
        logger.log(self.log_prefix + message)

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
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        kl_penalty_var = theano.shared(
            np.array(self.initial_kl_penalty,dtype=theano.config.floatX),
            name="kl_penalty"
        )

        if is_recurrent:
            raise NotImplementedError
        else:
            dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)

            lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
            ent = TT.mean(dist.entropy_sym(dist_info_vars))
            mean_kl = TT.mean(kl)

            clipped_lr = TT.clip(lr, 1. - self.clip_lr, 1. + self.clip_lr)

            surr_loss = - TT.mean(lr * advantage_var)
            clipped_surr_loss = - TT.mean(
                TT.min([lr * advantage_var, clipped_lr * advantage_var], axis=1)
            )

            clipped_surr_pen_loss = clipped_surr_loss - self.entropy_bonus_coeff * ent
            if self.use_kl_penalty:
                clipped_surr_pen_loss += kl_penalty_var * TT.maximum(0., mean_kl - self.step_size)
                # only penalize when the constraint is violated

            self.optimizer.update_opt(
                loss=clipped_surr_pen_loss,
                target=self.policy,
                inputs=[obs_var, action_var, advantage_var] + state_info_vars_list + old_dist_info_vars_list,
                diagnostic_vars=OrderedDict([
                    ("UnclippedSurrLoss", surr_loss),
                    ("MeanKL", mean_kl),
                ])
            )

        self._kl_penalty_var = kl_penalty_var

    def f_increase_penalty(self):
        new_value = np.minimum(
            self._kl_penalty_var.get_value() * self.increase_penalty_factor,
            self.max_penalty
        ).astype(theano.config.floatX)
        self._kl_penalty_var.set_value(new_value)
        return new_value

    def f_decrease_penalty(self):
        new_value = np.maximum(
            self._kl_penalty_var.get_value() * self.decrease_penalty_factor,
            self.min_penalty
        ).astype(theano.config.floatX)
        self._kl_penalty_var.set_value(new_value)
        return new_value

    def f_reset_penalty(self):
        self._kl_penalty_var.set_value(self.initial_kl_penalty)

    def sliced_loss_kl(self, inputs):
        """
        Compute loss and kl one minibatch at a time; then average the results.
        """
        loss, diags = self.optimizer.loss_diagnostics(inputs)
        return diags["UnclippedSurrLoss"], diags["MeanKL"]

    def optimize_policy(self, itr, samples_data):
        """
        For each iteration, outer loop is changing kl_penalty, while inner loop is calling a first-order optimizer to decrease (surr_loss + penalty * kl - entropy_bonus * entropy), one minibatch at a time, but all data are used in the inner loop. The outer loop is repeated at least self.min_n_epochs times; if the kl constraint is violated after that, increase kl penalty until the constraint is satisfied.
        """
        self.log("Policy param norm: %f" % np.linalg.norm(self.policy.get_param_values()))
        self.log("Start optimizing..")
        observations = samples_data["observations"]
        actions = samples_data["actions"]
        advantages = samples_data["advantages"]

        # Perform truncated backprop
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        all_inputs = [observations, actions, advantages] + state_info_list + dist_info_list
        if self.policy.recurrent:
            valids = samples_data["valids"]
            all_inputs.append(valids)

        # can be slow, as it uses all inputs
        if self.log_loss_kl_before:
            self.log("Computing loss / KL before training")
            surr_loss_before, kl_before = self.sliced_loss_kl(all_inputs)
            self.log("Computed")

        prev_params = self.policy.get_param_values(trainable=True)

        epoch_surr_losses = []
        epoch_mean_kls = []

        best_loss = None
        best_kl = None
        best_params = None

        self.f_reset_penalty()

        def itr_callback(itr, loss, diagnostics, *args, **kwargs):
            """
            Within self.min_n_epochs, if self.use_kl_penalty is used, increase / decrease penalty as kl too high / low. Whenever self.min_n_epochs is reached and kl is small, terminate. Otherwise, keep increasing kl penalty. Notice that optimizer.n_epochs also constrains the total number of iterations.
            :return True / False: continue / terminate optimization
            """
            nonlocal best_loss
            nonlocal best_params
            nonlocal best_kl
            surr_loss = diagnostics["UnclippedSurrLoss"]
            mean_kl = diagnostics["MeanKL"]
            epoch_surr_losses.append(surr_loss)
            epoch_mean_kls.append(mean_kl)

            # record the best params and diagnostics
            if mean_kl <= self.step_size:
                if best_loss is None or surr_loss < best_loss:
                    best_loss = surr_loss
                    best_kl = mean_kl
                    best_params = self.policy.get_param_values()
            if mean_kl <= self.step_size and itr + 1 >= self.min_n_epochs:
                penalty = self.f_decrease_penalty() # useful for next algo iteration
                self.log("Epoch %d; Loss %f; Mean KL: %f; decreasing penalty to %f and finish opt since KL and "
                           "minimum #epochs reached" % (itr, surr_loss, mean_kl, penalty))
                # early termination
                return False
            if self.use_kl_penalty:
                if mean_kl > self.step_size:
                    # constraint violated. increase penalty
                    penalty = self.f_increase_penalty()
                    self.log("Epoch %d; Loss %f; Mean KL: %f; increasing penalty to %f" % (
                        itr, surr_loss, mean_kl, penalty))
                else:
                    penalty = self.f_decrease_penalty()
                    self.log("Epoch %d; Loss %f; Mean KL: %f; decreasing penalty to %f" % (
                        itr, surr_loss, mean_kl, penalty))
            elif itr + 1 >= self.min_n_epochs:
                # if do not use kl penalty, only execute for the minimum number of epochs
                return False
            return True

        self.optimizer.optimize(all_inputs, callback=itr_callback)

        if best_params is not None:
            self.policy.set_param_values(best_params)

        # compute loss and kl constraint; backtrack if kl constraint violated
        if self.log_loss_kl_after or self.use_line_search:
            self.log("Computing loss / KL after training")
            surr_loss_after, kl_after = self.sliced_loss_kl(all_inputs)
            self.log("Computed")

            if self.use_line_search and kl_after > self.step_size:
                self.log("Performing line search to make sure KL is within range")
                n_trials = 0
                step_size = 1.
                now_params = self.policy.get_param_values(trainable=True)
                while kl_after > self.step_size and n_trials < self.max_backtracks:
                    step_size *= self.backtrack_ratio
                    self.policy.set_param_values(
                        (1 - step_size) * prev_params + step_size * now_params,
                        trainable=True
                    )
                    surr_loss_after, kl_after = self.sliced_loss_kl(all_inputs)
                    self.log("After shrinking step, loss = %f, Mean KL = %f" % (surr_loss_after, kl_after))

        # perform minibatch gradient descent on the surrogate loss, while monitoring the KL divergence

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
            if best_loss is None: # why would this be None?
                logger.record_tabular('LastEpoch.SurrLoss', epoch_surr_losses[-1])
                logger.record_tabular('LastEpoch.MeanKL', epoch_mean_kls[-1])
            else:
                logger.record_tabular('LastEpoch.SurrLoss', best_loss)
                logger.record_tabular('LastEpoch.MeanKL', best_kl)
        if self.log_loss_kl_before and self.log_loss_kl_after:
            logger.record_tabular('dSurrLoss', surr_loss_before - surr_loss_after)

        if np.isnan(epoch_surr_losses[-1]):
            self.log("NaN detected! Terminating")
            sys.exit()

        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
