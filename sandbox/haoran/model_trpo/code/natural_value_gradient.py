from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.haoran.model_trpo.code.utils import compute_analytic_value_gradient
from sandbox.haoran.model_trpo.code.utils import compute_approximate_value_gradient
from sandbox.haoran.model_trpo.code.utils import _worker_compute_jacobians

import theano
import theano.tensor as TT
import numpy as np

class NVG(BatchPolopt):
    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            truncate_local_is_ratio=None,
            fd_step=0,
            **kwargs
    ):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        # temporary
        assert isinstance(optimizer, ConjugateGradientOptimizer)
        self.optimizer = optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.fd_step = fd_step
        super(NVG, self).__init__(**kwargs)

    @overrides
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
            dtype=theano.config.floatX
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

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        flat_grad = TT.fvector('flat_grad')


        input_list = [
                 obs_var,
                 action_var,
                 advantage_var,
             ] + \
             state_info_vars_list + \
             old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        extra_inputs = (flat_grad,)
        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            extra_inputs=extra_inputs,
            constraint_name="mean_kl"
        )

        # override the conjugate_gradient_optimizer's loss gradient
        self.optimizer._opt_fun["f_grad"] = lambda: ext.compile_function(
            inputs=input_list + list(extra_inputs),
            outputs=flat_grad,
            log_name="f_grad_analytic",
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        paths = samples_data["paths"]


        # feed in gradients computed by recursion
        logger.log("Computing value gradients.")
        grads = []
        for i,path in enumerate(paths):
            # logger.log("Computing value gradient of path # %d."%(i))
            if self.fd_step > 1e-8:
                V_s, V_theta = compute_approximate_value_gradient(
                    self.env,
                    self.policy,
                    path,
                    self.discount,
                    self.fd_step,
                )
            else:
                V_s, V_theta = compute_analytic_value_gradient(
                    self.env,
                    self.policy,
                    path,
                    self.discount,
                )
            grads.append(np.copy(V_theta[0]))
        flat_grad = - np.average(np.asarray(grads),axis=0)
        extra_inputs = (flat_grad,)


        loss_before = self.optimizer.loss(all_input_values,extra_inputs)
        mean_kl_before = self.optimizer.constraint_val(all_input_values,extra_inputs)
        self.optimizer.optimize(all_input_values,extra_inputs)
        mean_kl = self.optimizer.constraint_val(all_input_values,extra_inputs)
        loss_after = self.optimizer.loss(all_input_values,extra_inputs)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
