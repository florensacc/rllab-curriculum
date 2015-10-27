import misc.logger as logger
from misc.ext import merge_dict
import numpy as np
from sampler import parallel_sampler
from policy import EpsilonGreedyPolicy
import scipy
from misc.tensor_utils import flatten_tensors
import misc.logger as logger

import cgtcompat as theano
import cgtcompat.tensor as T

def request_samples(eps_policy, epsilon, n_samples, max_path_length):
    eps_policy.epsilon = epsilon
    return parallel_sampler.request_samples(eps_policy.get_param_values(), n_samples, max_path_length)

def new_train_vars(qfunc):
    obs = qfunc.input_var
    actions = T.ivector("actions")
    rewards = T.vector("rewards")
    terminate = T.vector("terminate")
    penalty = T.scalar("penalty")
    prev_qval = T.matrix("prev_qval")
    return dict(
        obs=obs,
        actions=actions,
        rewards=rewards,
        terminate=terminate,
        prev_qval=prev_qval,
        penalty=penalty,
    )

def to_train_var_list(obs, actions, rewards, terminate, prev_qval, penalty):
    return [obs, actions, rewards, terminate, prev_qval, penalty]

def new_loss(qfunc, discount, obs, actions, rewards, terminate, prev_qval, penalty):
    qval = qfunc.qval_var
    N = obs.shape[0]
    qsa = qval[T.arange(N), actions]
    y = rewards + (1 - terminate) * discount * T.concatenate([T.max(prev_qval[1:], axis=1), np.array([0.0]).astype(theano.config.floatX)])
    prev_qsa = prev_qval[T.arange(N), actions]
    
    loss = T.mean(T.square(qsa - y))
    sigmasq = T.mean(T.square(prev_qsa - y))
    reg = T.mean(T.mean(T.square(qval - prev_qval)))
    reg_normalized = reg / (2*T.square(sigmasq))
    reg_loss = loss + penalty * reg
    return dict(
        loss=loss,
        reg=reg_normalized,
        reg_loss=reg_loss,
        sigmasq=sigmasq
    )

# Proximal Fitted Q-Iteration
class PFQI(object):

    def __init__(self,
            samples_per_itr=10000, max_epsilon=1, min_epsilon=0.1,
            epsilon_decay_range=100, discount=0.99, start_itr=0, n_itr=500,
            initial_penalty=1, max_opt_itr=50, test_samples_per_itr=10000,
            stepsize=0.1, adapt_penalty=True, max_penalty_itr=10,
            penalty_expand_factor=2, penalty_shrink_factor=0.5,
            max_path_length=np.inf):
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

    def train(self, mdp, qfunc):
        opt_info = self.init_opt(mdp, qfunc)
        self.start_worker(mdp, qfunc)
        for itr in xrange(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            samples_data = self.obtain_samples(itr, mdp, qfunc, opt_info)
            #for _ in range(10):
            opt_info = self.optimize_qfunc(itr, qfunc, samples_data, opt_info)
            self.test_performance(itr, qfunc)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

    def start_worker(self, mdp, qfunc):
        self.eps_policy = EpsilonGreedyPolicy(qfunc, epsilon=self.max_epsilon)
        parallel_sampler.populate_task(mdp, self.eps_policy)

    def init_opt(self, mdp, qfunc):
        train_vars = new_train_vars(qfunc)
        result_vars = new_loss(qfunc, self.discount, **train_vars)
        reg_loss_var = result_vars["reg_loss"]
        loss_var = result_vars["loss"]
        reg_var = result_vars["reg"]
        grads_var = T.grad(reg_loss_var, qfunc.params)
        train_var_list = to_train_var_list(**train_vars)
        logger.log("compiling functions...")
        f_loss = theano.function(train_var_list, [loss_var, reg_var, reg_loss_var], allow_input_downcast=True, on_unused_input='ignore')
        f_grads = theano.function(train_var_list, grads_var, allow_input_downcast=True, on_unused_input='ignore')
        return dict(
            f_loss=f_loss,
            f_grads=f_grads,
            penalty=self.initial_penalty,
        )

    def obtain_samples(self, itr, mdp, qfunc, opt_info):
        epsilon = max(self.min_epsilon, self.max_epsilon - (self.max_epsilon - self.min_epsilon) * itr / self.epsilon_decay_range)
        logger.record_tabular("Epsilon", epsilon)
        paths = request_samples(self.eps_policy, epsilon, self.samples_per_itr, self.max_path_length)
        observations = np.vstack([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"].reshape(-1) for path in paths])
        rewards = np.concatenate([path["rewards"].reshape(-1) for path in paths])
        terminate = np.concatenate([np.append(np.zeros(len(path["rewards"]) - 1), 1) for path in paths]).astype(int)
        prev_qval = qfunc.compute_qval(observations)
        train_vals = [observations, actions, rewards, terminate, prev_qval]

        logger.record_tabular('MaxQAbs', np.max(np.abs(prev_qval.reshape(-1))))
        logger.record_tabular('SampleRewMean', np.mean([sum(path["rewards"]) for path in paths]))
        return dict(
            train_vals=train_vals,
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminate=terminate,
            paths=paths,
            prev_qval=prev_qval,
        )

    def optimize_qfunc(self, itr, qfunc, samples_data, opt_info):
        train_vals = samples_data['train_vals']
        f_loss = opt_info['f_loss']
        f_grads = opt_info['f_grads']
        penalty = opt_info['penalty']

        def evaluate_loss(train_vals):
            def evaluate(params):
                qfunc.set_param_values(params)
                loss, reg, reg_loss = f_loss(*train_vals)
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
        reg = None
        penalty_scale_factor = None

        argmax_indices = samples_data['prev_qval'].argmax(axis=1)

        logger.record_tabular('LossBefore', loss_before)

        for penalty_itr in range(self.max_penalty_itr):
            result = scipy.optimize.fmin_l_bfgs_b(
                func=evaluate_loss(train_vals + [try_penalty]),
                x0=cur_params,
                fprime=evaluate_grad(train_vals + [try_penalty]),
                maxiter=self.max_opt_itr
            )
            try_loss, try_reg, _ = f_loss(*(train_vals + [try_penalty]))

            new_argmax_indices = qfunc.compute_qval(samples_data['observations']).argmax(axis=1)
            n_actions_changed = np.sum(argmax_indices != new_argmax_indices)
            logger.log('penalty %f => loss %f, reg %f, #actions changed %d' % (try_penalty, try_loss, try_reg, n_actions_changed))

            if try_reg < self.stepsize or \
                    (penalty_itr == self.max_penalty_itr - 1 and opt_params is None) or \
                    not self.adapt_penalty:
                opt_params = result[0]
                penalty = try_penalty
                reg = try_reg

            if not self.adapt_penalty:
                break

            # decide scale factor on the first iteration
            if penalty_scale_factor is None or np.isnan(try_reg):
                if try_reg > self.stepsize or np.isnan(try_reg):
                    # need to increase penalty
                    penalty_scale_factor = self.penalty_expand_factor
                else:
                    # can shrink penalty
                    penalty_scale_factor = self.penalty_shrink_factor
            else:
                if penalty_scale_factor > 1 and try_reg <= self.stepsize:
                    break
                elif penalty_scale_factor < 1 and try_reg >= self.stepsize:
                    break
            try_penalty *= penalty_scale_factor

        loss_after, _, _ = f_loss(*(train_vals + [penalty]))

        logger.record_tabular('LossAfter', loss_after)
        qfunc.set_param_values(opt_params)
        return merge_dict(opt_info, dict(
            cur_params=cur_params,
            opt_params=opt_params,
            penalty=penalty
        ))

    def test_performance(self, itr, qfunc):
        # test performance
        test_paths = request_samples(self.eps_policy, 0, self.test_samples_per_itr, self.max_path_length)
        avg_reward = np.mean([sum(path["rewards"]) for path in test_paths])
        logger.record_tabular('TestRewMean', avg_reward)
