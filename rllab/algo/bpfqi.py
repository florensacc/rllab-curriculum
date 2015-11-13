import misc.logger as logger
from misc.special import discount_cumsum, cat_perplexity
from misc.ext import merge_dict
import numpy as np
from sampler import parallel_sampler
from policy import BoltzmannPolicy
import scipy
from misc.tensor_utils import flatten_tensors
import misc.logger as logger

import tensorfuse as theano
import tensorfuse.tensor as T

def request_samples(boltz_policy, n_samples, max_path_length):
    return parallel_sampler.request_samples(boltz_policy.get_param_values(), n_samples, max_path_length)

def new_train_vars(qfunc, boltz_policy):
    obs = qfunc.input_var
    actions = T.ivector("actions")
    rewards = T.vector("rewards")
    terminate = T.vector("terminate")
    prev_pdist = T.matrix("prev_pdist")
    prev_qval = T.matrix("prev_qval")
    emp_qval = T.vector("emp_qval")
    penalty = T.scalar("penalty")
    return dict(
        obs=obs,
        actions=actions,
        rewards=rewards,
        terminate=terminate,
        prev_pdist=prev_pdist,
        prev_qval=prev_qval,
        emp_qval=emp_qval,
        penalty=penalty,
    )

def to_train_var_list(obs, actions, rewards, terminate, prev_pdist, prev_qval, emp_qval, penalty):
    return [obs, actions, rewards, terminate, prev_pdist, prev_qval, emp_qval, penalty]


# 1/N * sum pi(a|s) / pi_old(a|s) * (-emp_qval) + 1/N * sum (Qi - yi)^2 + penalty*KL
# Some variants to try out:
#   - Use q value estimates as opposed to Monte-Carlo Q estimates
#   - Use a per-state temperature
def new_loss(qfunc, boltz_policy, discount, obs, actions, rewards, terminate, prev_pdist, prev_qval, emp_qval, penalty):
    qval = qfunc.qval_var
    pdist = boltz_policy.pdist_var

    N = obs.shape[0]
    qsa = qval[T.arange(N), actions]
    prev_qsa = prev_qval[T.arange(N), actions]
    # Here, we have a natural choice of baseline: V(s) = max_a Q(s, a)
    emp_aval = emp_qval - T.max(prev_qval, axis=1)

    y = rewards + (1 - terminate) * discount * T.concatenate([T.max(prev_qval[1:], axis=1), np.array([0.0]).astype(theano.config.floatX)])

    lr = boltz_policy.likelihood_ratio(prev_pdist, pdist, actions)
    mean_kl = T.mean(boltz_policy.kl(prev_pdist, pdist))

    policy_loss = T.mean(lr * (-emp_qval))
    qfunc_loss = T.mean(T.square(qsa - y))
    policy_reg_loss = policy_loss + penalty*mean_kl
    qfunc_reg_loss = qfunc_loss + penalty*mean_kl
    joint_loss = policy_loss + qfunc_loss + penalty*mean_kl
    
    return dict(
        policy_loss=policy_loss,
        qfunc_loss=qfunc_loss,
        policy_reg_loss=policy_reg_loss,
        qfunc_reg_loss=qfunc_reg_loss,
        mean_kl=mean_kl,
        joint_loss=joint_loss,
    )

#def new_policy_loss(

# Boltzman Proximal Fitted Q-Iteration
# Here, we form a policy using the formula for Boltzman exploration
class BPFQI(object):

    def __init__(self,
            samples_per_itr=10000, max_epsilon=1, min_epsilon=0.1,
            epsilon_decay_range=20, discount=0.99, start_itr=0, n_itr=100,
            initial_penalty=1, max_opt_itr=50, test_samples_per_itr=10000,
            stepsize=0.1, adapt_penalty=True, max_penalty_itr=10,
            penalty_expand_factor=2, penalty_shrink_factor=0.5,
            max_path_length=np.inf, initial_temperature=None,
            learn_temperature_threshold=0.1, temperature_samples=10000,
            opt_mode='separate'):
        self.samples_per_itr = samples_per_itr
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_range = epsilon_decay_range
        self.discount = discount
        self.start_itr = start_itr
        self.n_itr = n_itr
        self.initial_penalty = initial_penalty
        self.max_opt_itr = max_opt_itr
        self.test_samples_per_itr = test_samples_per_itr
        self.stepsize = stepsize
        self.adapt_penalty = adapt_penalty
        self.max_penalty_itr = max_penalty_itr
        self.penalty_expand_factor = penalty_expand_factor
        self.penalty_shrink_factor = penalty_shrink_factor
        self.max_path_length = max_path_length
        self.initial_temperature = initial_temperature
        self.temperature_samples = temperature_samples
        self.learn_temperature_threshold = learn_temperature_threshold
        self.opt_mode = opt_mode

    def train(self, mdp, qfunc):
        boltz_policy = BoltzmannPolicy(qfunc, temperature=self.initial_temperature)
        self.start_worker(mdp, qfunc, boltz_policy)
        opt_info = self.init_opt(mdp, qfunc, boltz_policy)
        for itr in xrange(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            samples_data = self.obtain_samples(itr, mdp, qfunc, boltz_policy, opt_info)
            if self.opt_mode == 'joint':
                opt_info = self.joint_optimize(itr, qfunc, boltz_policy, samples_data, opt_info)
            else:
                opt_info = self.optimize_qfunc(itr, qfunc, boltz_policy, samples_data, opt_info)
                opt_info = self.optimize_policy(itr, qfunc, boltz_policy, samples_data, opt_info)
            self.test_performance(itr, boltz_policy)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

    def start_worker(self, mdp, qfunc, boltz_policy):
        parallel_sampler.populate_task(mdp, boltz_policy)

    def init_opt(self, mdp, qfunc, boltz_policy):
        train_vars = new_train_vars(qfunc, boltz_policy)
        result_vars = new_loss(qfunc, boltz_policy, self.discount, **train_vars)
        policy_loss_var = result_vars["policy_loss"]
        qfunc_loss_var = result_vars["qfunc_loss"]
        mean_kl_var = result_vars["mean_kl"]

        if self.opt_mode == 'joint':
            joint_loss_var = result_vars["joint_loss"]
            grads_var = T.grad(joint_loss_var, boltz_policy.params)
        else:
            policy_reg_loss_var = result_vars["policy_reg_loss"]
            qfunc_reg_loss_var = result_vars["qfunc_reg_loss"]
            # Here, we only optimize the temperature when optimizing the policy
            policy_grads_var = T.grad(policy_reg_loss_var, boltz_policy.self_params)
            # The temperature is held fixed when performing Q-iteration
            qfunc_grads_var = T.grad(qfunc_reg_loss_var, qfunc.params)

        train_var_list = to_train_var_list(**train_vars)

        if self.initial_temperature is None:
            logger.log("collecting samples for estimating initial temperature...")
            # First, we collect some samples and figure out what the initial temperature should be
            paths = request_samples(boltz_policy, self.temperature_samples, self.max_path_length)
            observations = np.vstack([path["observations"] for path in paths])
            qval = qfunc.compute_qval(observations)
            # learn temperature so that the average perplexity is 
            boltz_policy.learn_temperature(qval, self.learn_temperature_threshold)
            logger.log("learned temperature: %f" % boltz_policy.temperature)

        logger.log("compiling functions...")
        if self.opt_mode == 'joint':
            f_loss = theano.function(train_var_list, [policy_loss_var, qfunc_loss_var, mean_kl_var, joint_loss_var], allow_input_downcast=True, on_unused_input='ignore')
            f_grads = theano.function(train_var_list, grads_var, allow_input_downcast=True, on_unused_input='ignore')
            return dict(
                f_loss=f_loss,
                f_grads=f_grads,
                penalty=self.initial_penalty,
            )
        else:
            f_policy_loss = theano.function(train_var_list, [policy_loss_var, mean_kl_var, policy_reg_loss_var], allow_input_downcast=True, on_unused_input='ignore')
            f_policy_grads = theano.function(train_var_list, policy_grads_var, allow_input_downcast=True, on_unused_input='ignore')
            f_qfunc_loss = theano.function(train_var_list, [qfunc_loss_var, mean_kl_var, qfunc_reg_loss_var], allow_input_downcast=True, on_unused_input='ignore')
            f_qfunc_grads = theano.function(train_var_list, qfunc_grads_var, allow_input_downcast=True, on_unused_input='ignore')
            return dict(
                f_policy_loss=f_policy_loss,
                f_policy_grads=f_policy_grads,
                f_qfunc_loss=f_qfunc_loss,
                f_qfunc_grads=f_qfunc_grads,
                policy_penalty=self.initial_penalty,
                qfunc_penalty=self.initial_penalty,
            )

    def obtain_samples(self, itr, mdp, qfunc, boltz_policy, opt_info):
        paths = request_samples(boltz_policy, self.samples_per_itr, self.max_path_length)
        observations = np.vstack([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"].reshape(-1) for path in paths])
        rewards = np.concatenate([path["rewards"].reshape(-1) for path in paths])

        terminate = np.concatenate([np.append(np.zeros(len(path["rewards"]) - 1), 1) for path in paths]).astype(int)
        prev_pdists = np.vstack([path["pdists"] for path in paths])
        prev_qval = qfunc.compute_qval(observations)
        emp_qval = np.concatenate([discount_cumsum(path["rewards"], self.discount) for path in paths])
        train_vals = [observations, actions, rewards, terminate, prev_pdists, prev_qval, emp_qval]

        logger.record_tabular('MaxQAbs', np.max(np.abs(prev_qval.reshape(-1))))
        logger.record_tabular('SampleRewMean', np.mean([sum(path["rewards"]) for path in paths]))
        logger.record_tabular('Perplexity', np.mean(cat_perplexity(prev_pdists)))
        return dict(
            train_vals=train_vals,
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminate=terminate,
            paths=paths,
            prev_pdists=prev_pdists,
            prev_qval=prev_qval,
            emp_qval=emp_qval,
        )

    def joint_optimize(self, itr, qfunc, boltz_policy, samples_data, opt_info):
        train_vals = samples_data['train_vals']
        f_loss = opt_info['f_loss']
        f_grads = opt_info['f_grads']
        penalty = opt_info['penalty']

        def evaluate_loss(train_vals):
            def evaluate(params):
                boltz_policy.set_param_values(params)
                policy_loss, qfunc_loss, mean_kl, joint_loss = f_loss(*train_vals)
                return joint_loss.astype(np.float64)
            return evaluate

        def evaluate_grad(train_vals):
            def evaluate(params):
                boltz_policy.set_param_values(params)
                grad = f_grads(*train_vals)
                flattened_grad = flatten_tensors(map(np.asarray, grad))
                return flattened_grad.astype(np.float64)
            return evaluate

        policy_loss_before, qfunc_loss_before, _, _ = f_loss(*(train_vals + [0]))
        loss_before = policy_loss_before + qfunc_loss_before

        cur_params = boltz_policy.get_param_values()

        try_penalty = np.clip(penalty, 1e-2, 1e6)

        opt_params = None
        final_mean_kl = None
        penalty_scale_factor = None

        argmax_indices = samples_data['prev_qval'].argmax(axis=1)

        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('TempBefore', boltz_policy.temperature)

        for penalty_itr in range(self.max_penalty_itr):
            result = scipy.optimize.fmin_l_bfgs_b(
                func=evaluate_loss(train_vals + [try_penalty]),
                x0=cur_params,
                fprime=evaluate_grad(train_vals + [try_penalty]),
                maxiter=self.max_opt_itr
            )
            try_policy_loss, try_qfunc_loss, try_mean_kl, try_joint_loss = f_loss(*(train_vals + [try_penalty]))

            logger.log('penalty %f => policy loss %f, qval loss %f, mean kl %f' % (try_penalty, try_policy_loss, try_qfunc_loss, try_mean_kl))

            if try_mean_kl < self.stepsize or \
                    (penalty_itr == self.max_penalty_itr - 1 and opt_params is None) or \
                    not self.adapt_penalty:
                opt_params = result[0]
                penalty = try_penalty
                final_mean_kl = try_mean_kl

            if not self.adapt_penalty:
                break

            # decide scale factor on the first iteration
            if penalty_scale_factor is None or np.isnan(try_mean_kl):
                if try_mean_kl > self.stepsize or np.isnan(try_mean_kl):
                    # need to increase penalty
                    penalty_scale_factor = self.penalty_expand_factor
                else:
                    # can shrink penalty
                    penalty_scale_factor = self.penalty_shrink_factor
            else:
                if penalty_scale_factor > 1 and try_mean_kl <= self.stepsize:
                    break
                elif penalty_scale_factor < 1 and try_mean_kl >= self.stepsize:
                    break
            try_penalty *= penalty_scale_factor

        policy_loss_after, qfunc_loss_after, _, _ = f_loss(*(train_vals + [penalty]))
        loss_after = policy_loss_after + qfunc_loss_after

        boltz_policy.set_param_values(opt_params)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('TempAfter', boltz_policy.temperature)
        return merge_dict(opt_info, dict(
            cur_params=cur_params,
            opt_params=opt_params,
            penalty=penalty
        ))

    def optimize_qfunc(self, itr, qfunc, boltz_policy, samples_data, opt_info):
        train_vals = samples_data['train_vals']
        f_loss = opt_info['f_qfunc_loss']
        f_grads = opt_info['f_qfunc_grads']
        penalty = opt_info['qfunc_penalty']

        def evaluate_loss(train_vals):
            def evaluate(params):
                qfunc.set_param_values(params)
                loss, mean_kl, reg_loss = f_loss(*train_vals)
                return reg_loss.astype(np.float64)
            return evaluate

        def evaluate_grad(train_vals):
            def evaluate(params):
                qfunc.set_param_values(params)
                grad = f_grads(*train_vals)
                flattened_grad = flatten_tensors(map(np.asarray, grad))
                return flattened_grad.astype(np.float64)
            return evaluate

        loss_before, _, _ = f_loss(*(train_vals + [0]))

        cur_params = qfunc.get_param_values()

        try_penalty = np.clip(penalty, 1e-2, 1e6)

        opt_params = None
        final_mean_kl = None
        penalty_scale_factor = None

        logger.record_tabular('QfuncLossBefore', loss_before)

        for penalty_itr in range(self.max_penalty_itr):
            result = scipy.optimize.fmin_l_bfgs_b(
                func=evaluate_loss(train_vals + [try_penalty]),
                x0=cur_params,
                fprime=evaluate_grad(train_vals + [try_penalty]),
                maxiter=self.max_opt_itr
            )
            try_loss, try_mean_kl, try_reg_loss = f_loss(*(train_vals + [try_penalty]))

            logger.log('penalty %f => qfunc loss %f, mean kl %f' % (try_penalty, try_loss, try_mean_kl))

            if try_mean_kl < self.stepsize or \
                    (penalty_itr == self.max_penalty_itr - 1 and opt_params is None) or \
                    not self.adapt_penalty:
                opt_params = result[0]
                penalty = try_penalty
                final_mean_kl = try_mean_kl

            if not self.adapt_penalty:
                break

            # decide scale factor on the first iteration
            if penalty_scale_factor is None or np.isnan(try_mean_kl):
                if try_mean_kl > self.stepsize or np.isnan(try_mean_kl):
                    # need to increase penalty
                    penalty_scale_factor = self.penalty_expand_factor
                else:
                    # can shrink penalty
                    penalty_scale_factor = self.penalty_shrink_factor
            else:
                if penalty_scale_factor > 1 and try_mean_kl <= self.stepsize:
                    break
                elif penalty_scale_factor < 1 and try_mean_kl >= self.stepsize:
                    break
            try_penalty *= penalty_scale_factor

        loss_after, _, _ = f_loss(*(train_vals + [penalty]))

        qfunc.set_param_values(opt_params)
        logger.record_tabular('QfuncLossAfter', loss_after)
        return merge_dict(opt_info, dict(
            qfunc_cur_params=cur_params,
            qfunc_opt_params=opt_params,
            qfunc_penalty=penalty
        ))

    def optimize_policy(self, itr, qfunc, boltz_policy, samples_data, opt_info):
        train_vals = samples_data['train_vals']
        f_loss = opt_info['f_policy_loss']
        f_grads = opt_info['f_policy_grads']
        penalty = opt_info['policy_penalty']

        def evaluate_loss(train_vals):
            def evaluate(params):
                boltz_policy.set_self_param_values(params)
                loss, mean_kl, reg_loss = f_loss(*train_vals)
                return reg_loss.astype(np.float64)
            return evaluate

        def evaluate_grad(train_vals):
            def evaluate(params):
                boltz_policy.set_self_param_values(params)
                grad = f_grads(*train_vals)
                flattened_grad = flatten_tensors(map(np.asarray, grad))
                return flattened_grad.astype(np.float64)
            return evaluate

        loss_before, _, _ = f_loss(*(train_vals + [0]))

        cur_params = boltz_policy.get_self_param_values()

        try_penalty = np.clip(penalty, 1e-2, 1e6)

        opt_params = None
        final_mean_kl = None
        penalty_scale_factor = None

        logger.record_tabular('PolicyLossBefore', loss_before)

        for penalty_itr in range(self.max_penalty_itr):
            result = scipy.optimize.fmin_l_bfgs_b(
                func=evaluate_loss(train_vals + [try_penalty]),
                x0=cur_params,
                fprime=evaluate_grad(train_vals + [try_penalty]),
                maxiter=self.max_opt_itr
            )
            try_loss, try_mean_kl, try_reg_loss = f_loss(*(train_vals + [try_penalty]))

            logger.log('penalty %f => policy loss %f, mean kl %f' % (try_penalty, try_loss, try_mean_kl))

            if try_mean_kl < self.stepsize or \
                    (penalty_itr == self.max_penalty_itr - 1 and opt_params is None) or \
                    not self.adapt_penalty:
                opt_params = result[0]
                penalty = try_penalty
                final_mean_kl = try_mean_kl

            if not self.adapt_penalty:
                break

            # decide scale factor on the first iteration
            if penalty_scale_factor is None or np.isnan(try_mean_kl):
                if try_mean_kl > self.stepsize or np.isnan(try_mean_kl):
                    # need to increase penalty
                    penalty_scale_factor = self.penalty_expand_factor
                else:
                    # can shrink penalty
                    penalty_scale_factor = self.penalty_shrink_factor
            else:
                if penalty_scale_factor > 1 and try_mean_kl <= self.stepsize:
                    break
                elif penalty_scale_factor < 1 and try_mean_kl >= self.stepsize:
                    break
            try_penalty *= penalty_scale_factor

        loss_after, _, _ = f_loss(*(train_vals + [penalty]))

        boltz_policy.set_self_param_values(opt_params)
        logger.record_tabular('PolicyLossAfter', loss_after)
        return merge_dict(opt_info, dict(
            policy_cur_params=cur_params,
            policy_opt_params=opt_params,
            policy_penalty=penalty
        ))

    def test_performance(self, itr, boltz_policy):
        # test performance
        cur_temp = boltz_policy.temperature
        boltz_policy.temperature = cur_temp / 1000
        test_paths = request_samples(boltz_policy, self.test_samples_per_itr, self.max_path_length)
        boltz_policy.temperature = cur_temp 
        avg_reward = np.mean([sum(path["rewards"]) for path in test_paths])
        logger.record_tabular('TestRewMean', avg_reward)
