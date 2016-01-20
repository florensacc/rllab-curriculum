from rllab.misc.tensor_utils import flatten_tensors
from rllab.misc.ext import merge_dict, compile_function, extract, new_tensor
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.algo.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from pydoc import locate
import numpy as np


class PPO(BatchPolopt):
    """
    Proximal Policy Optimization.
    """

    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.arg("step_size", type=float,
                  help="Maximum change in mean KL per iteration.")
    @autoargs.arg("initial_penalty", type=float,
                  help="Initial value of the penalty coefficient.")
    @autoargs.arg("min_penalty", type=float,
                  help="Minimum value of penalty coefficient.")
    @autoargs.arg("max_penalty", type=float,
                  help="Maximum value of penalty coefficient.")
    @autoargs.arg("increase_penalty_factor", type=float,
                  help="How much the penalty should increase if kl divergence "
                       "exceeds the threshold on the first penalty "
                       "iteration.")
    @autoargs.arg("decrease_penalty_factor", type=float,
                  help="How much the penalty should decrease if kl divergence "
                       "is less than the threshold on the first penalty "
                       "iteration.")
    @autoargs.arg("max_opt_itr", type=int,
                  help="Maximum number of batch optimization iterations.")
    @autoargs.arg("max_penalty_itr", type=int,
                  help="Maximum number of penalty iterations.")
    @autoargs.arg("binary_search_penalty", type=bool,
                  help="Whether to search for more precise penalty after the "
                       "initial exponentially adjusted value.")
    @autoargs.arg("max_penalty_bs_itr", type=int,
                  help="Maximum number of binary search iterations.")
    @autoargs.arg("bs_kl_tolerance", type=float,
                  help="Tolerance level for binary search.")
    @autoargs.arg("adapt_penalty", type=bool,
                  help="Whether to adjust penalty for each iteration.")
    @autoargs.arg("optimizer", type=str,
                  help="Module path to the optimizer. It must support the "
                       "same interface as scipy.optimize.fmin_l_bfgs_b")
    def __init__(
            self,
            step_size=0.01,
            initial_penalty=1,
            min_penalty=1e-2,
            max_penalty=1e6,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            max_opt_itr=20,
            max_penalty_itr=10,
            binary_search_penalty=True,
            max_penalty_bs_itr=10,
            bs_kl_tolerance=1e-3,
            adapt_penalty=True,
            optimizer='scipy.optimize.fmin_l_bfgs_b',
            **kwargs):
        self.step_size = step_size
        self.initial_penalty = initial_penalty
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.max_opt_itr = max_opt_itr
        self.max_penalty_itr = max_penalty_itr
        self.binary_search_penalty = binary_search_penalty
        self.max_penalty_bs_itr = max_penalty_bs_itr
        self.bs_kl_tolerance = bs_kl_tolerance
        self.adapt_penalty = adapt_penalty
        self.optimizer = locate(optimizer)
        super(PPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline):
        input_var = new_tensor(
            'input',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.vector('advantage')
        old_pdist_var = TT.matrix('old_pdist')
        action_var = TT.matrix('action', dtype=mdp.action_dtype)
        penalty_var = TT.scalar('penalty')

        pdist_var = policy.get_pdist_sym(input_var)
        kl = policy.kl(old_pdist_var, pdist_var)
        lr = policy.likelihood_ratio(old_pdist_var, pdist_var, action_var)
        mean_kl = TT.mean(kl)
        # formulate as a minimization problem
        surr_loss = - TT.mean(lr * advantage_var)
        surr_obj = surr_loss + penalty_var * mean_kl

        input_list = [
            input_var,
            advantage_var,
            old_pdist_var,
            action_var,
            penalty_var
        ]

        grads = theano.gradient.grad(
            surr_obj, policy.get_params(trainable=True))
        f_surr_kl = compile_function(
            input_list, [surr_obj, surr_loss, mean_kl])
        f_grads = compile_function(input_list, grads)
        penalty = self.initial_penalty

        return dict(
            f_surr_kl=f_surr_kl,
            f_grads=f_grads,
            penalty=penalty,
        )

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        penalty = opt_info['penalty']
        f_surr_kl = opt_info['f_surr_kl']
        f_grads = opt_info['f_grads']
        all_input_values = list(extract(
            samples_data,
            "observations", "advantages", "pdists", "actions"
        ))

        cur_params = policy.get_param_values(trainable=True)

        def evaluate_cost(penalty):
            def evaluate(params):
                policy.set_param_values(params, trainable=True)
                inputs_with_penalty = all_input_values + [penalty]
                val, _, _ = f_surr_kl(*inputs_with_penalty)
                return val.astype(np.float64)
            return evaluate

        def evaluate_grad(penalty):
            def evaluate(params):
                policy.set_param_values(params, trainable=True)
                grad = f_grads(*(all_input_values + [penalty]))
                flattened_grad = flatten_tensors(map(np.asarray, grad))
                return flattened_grad.astype(np.float64)
            return evaluate

        loss_before = evaluate_cost(0)(cur_params)
        logger.record_tabular('LossBefore', loss_before)

        try_penalty = np.clip(penalty, self.min_penalty, self.max_penalty)

        # search for the best penalty parameter
        penalty_scale_factor = None
        opt_params = None
        max_penalty_itr = self.max_penalty_itr
        mean_kl = None
        search_succeeded = False
        for penalty_itr in range(max_penalty_itr):
            logger.log('trying penalty=%.3f...' % try_penalty)
            self.optimizer(
                func=evaluate_cost(try_penalty), x0=cur_params,
                fprime=evaluate_grad(try_penalty),
                maxiter=self.max_opt_itr
                )
            _, try_loss, try_mean_kl = f_surr_kl(
                *(all_input_values + [try_penalty]))
            logger.log('penalty %f => loss %f, mean kl %f' %
                       (try_penalty, try_loss, try_mean_kl))
            if try_mean_kl < self.step_size or \
                    (penalty_itr == max_penalty_itr - 1 and
                     opt_params is None):
                opt_params = policy.get_param_values(trainable=True)
                penalty = try_penalty
                mean_kl = try_mean_kl

            if not self.adapt_penalty:
                break

            # decide scale factor on the first iteration
            if penalty_scale_factor is None or np.isnan(try_mean_kl):
                if try_mean_kl > self.step_size or np.isnan(try_mean_kl):
                    # need to increase penalty
                    penalty_scale_factor = self.increase_penalty_factor
                else:
                    # can shrink penalty
                    penalty_scale_factor = self.decrease_penalty_factor
            else:
                if penalty_scale_factor > 1 and \
                        try_mean_kl <= self.step_size:
                    search_succeeded = True
                    break
                elif penalty_scale_factor < 1 and \
                        try_mean_kl >= self.step_size:
                    search_succeeded = True
                    break
            try_penalty *= penalty_scale_factor
            if try_penalty < self.min_penalty or \
                    try_penalty > self.max_penalty:
                try_penalty = np.clip(
                    try_penalty, self.min_penalty, self.max_penalty)
                opt_params = policy.get_param_values(trainable=True)
                penalty = try_penalty
                mean_kl = try_mean_kl
                break

        if self.adapt_penalty and self.binary_search_penalty and \
                search_succeeded:
            # perform more fine-grained search of penalty parameter
            logger.log('Perform binary search for fine grained penalty')
            if penalty_scale_factor > 1:
                min_bs_penalty = try_penalty / penalty_scale_factor
                max_bs_penalty = try_penalty
            else:
                min_bs_penalty = try_penalty
                max_bs_penalty = try_penalty / penalty_scale_factor
            logger.log('Min penalty: %f; max penalty: %f' %
                       (min_bs_penalty, max_bs_penalty))
            for _ in range(self.max_penalty_bs_itr):
                penalty = 0.5 * (min_bs_penalty + max_bs_penalty)
                self.optimizer(
                    func=evaluate_cost(penalty), x0=cur_params,
                    fprime=evaluate_grad(penalty),
                    maxiter=self.max_opt_itr
                    )
                _, loss_after, mean_kl = f_surr_kl(
                    *(all_input_values + [penalty]))
                logger.log('penalty %f => loss %f, mean kl %f' %
                           (penalty, loss_after, mean_kl))
                if abs(mean_kl - self.step_size) < self.bs_kl_tolerance:
                    break
                if mean_kl > self.step_size:
                    # need to increase penalty
                    min_bs_penalty = penalty
                else:
                    max_bs_penalty = penalty
            opt_params = policy.get_param_values(trainable=True)

        policy.set_param_values(opt_params, trainable=True)
        loss_after = evaluate_cost(0)(opt_params)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return merge_dict(opt_info, dict(penalty=penalty))

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data,
                         opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )
