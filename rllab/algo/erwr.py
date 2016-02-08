from rllab.misc.tensor_utils import flatten_tensors
from rllab.misc.ext import merge_dict, compile_function, extract, new_tensor, \
    flatten_tensor_variables, unflatten_tensor_variables
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.algo.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from pydoc import locate
import numpy as np


class ERWR(BatchPolopt):
    """
    Episodic Reward Weighted Regression [1]_

    Notes
    -----
    This does not implement the original RwR [2]_ that deals with "immediate reward problems" since
    it doesn't find solutions that optimize for temporally delayed rewards.

    .. [1] Kober, Jens, and Jan R. Peters. "Policy search for motor primitives in robotics." Advances in neural information processing systems. 2009.
    .. [2] Peters, Jan, and Stefan Schaal. "Using reward-weighted regression for reinforcement learning of task space control." Approximate Dynamic Programming and Reinforcement Learning, 2007. ADPRL 2007. IEEE International Symposium on. IEEE, 2007.
    """

    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.arg("best_quantile", type=float,
                  help="The best quantile to use in training")
    @autoargs.arg("max_opt_itr", type=int,
                  help="Maximum number of batch optimization iterations.")
    @autoargs.arg("optimizer", type=str,
                  help="Module path to the optimizer. It must support the "
                       "same interface as scipy.optimize.fmin_l_bfgs_b")
    def __init__(
            self,
            best_quantile=1.,
            max_opt_itr=50,
            optimizer='scipy.optimize.fmin_l_bfgs_b',
            **kwargs):
        assert best_quantile == 1., "not implemented"
        self.best_quantile = best_quantile
        self.max_opt_itr = max_opt_itr
        self.optimizer = locate(optimizer)
        super(ERWR, self).__init__(**kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline):
        input_var = new_tensor(
            'input',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.vector('advantage')
        action_var = TT.matrix('action', dtype=mdp.action_dtype)

        # formulate as a minimization problem
        lower_bound = - TT.mean(advantage_var * policy.get_log_prob_sym(input_var, action_var))

        input_list = [
            input_var,
            advantage_var,
            action_var,
        ]

        trainable_params = policy.get_params(trainable=True)
        grads = theano.gradient.grad(
            lower_bound, trainable_params)
        f_lb = compile_function(
            input_list, lower_bound)

        param_var = TT.vector('flat param')
        opt_input_list = input_list + [param_var]
        param_vars = unflatten_tensor_variables(
            param_var, policy.get_param_shapes(trainable=True), trainable_params)
        flat_grad = flatten_tensor_variables(grads)
        f_opt = compile_function(
            opt_input_list,
            theano.clone(
                output=[lower_bound, flat_grad],
                replace=zip(trainable_params, param_vars)
            )
        )

        old_pdist_var = TT.matrix('old_pdist')
        pdist_var = policy.get_pdist_sym(input_var)
        kl = policy.kl(old_pdist_var, pdist_var)
        mean_kl = TT.mean(kl)
        max_kl = TT.max(kl)
        f_kl = compile_function(
            inputs=input_list + [old_pdist_var],
            outputs=[mean_kl, max_kl],
        )

        return dict(
            f_lb=f_lb,
            f_opt=f_opt,
            f_kl=f_kl,
        )

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        f_lb, f_opt = extract(opt_info, "f_lb", "f_opt")
        full_len_input_values = list(extract(
            samples_data,
            "observations", "advantages", "actions"
        ))
        input_values = [
            val[:int(self.best_quantile * val.shape[0])] for val in full_len_input_values
        ]

        cur_params = policy.get_param_values(trainable=True)
        loss_before = f_lb(*input_values)
        logger.record_tabular('LossBefore', loss_before)
        wrapped_func = lambda flat_param: [v.astype('float64') for v in f_opt(*(input_values + [flat_param]))]
        opt_params, loss_after = self.optimizer(
            func=wrapped_func,
            x0=cur_params,
            maxiter=self.max_opt_itr
        )[:2]
        policy.set_param_values(opt_params, trainable=True)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)

        mean_kl, max_kl = opt_info['f_kl'](*(list(full_len_input_values) + [samples_data['pdists']]))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data,
                         opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )

