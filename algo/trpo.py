import theano
import theano.tensor as T
import numpy as np
import scipy.optimize
import lasagne.layers as L
import operator
import sys
from misc.logging import Message
from misc.tensor_utils import flatten_tensors, unflatten_tensors
from collections import defaultdict

def reduce_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b

def reduce_mul(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a * b

class TRPO(object):

    def __init__(self, n_itr=500, samples_per_itr=100000, discount=0.99,
            stepsize=0.015, initial_lambda=1, max_opt_itr=100, 
            adapt_lambda=True, reuse_lambda=True):
        self.n_itr = n_itr
        self.samples_per_itr = samples_per_itr
        self.discount = discount
        self.stepsize = stepsize
        self.initial_lambda = initial_lambda
        self.max_opt_itr = max_opt_itr
        self.adapt_lambda = adapt_lambda
        # whether to start from the currently adapted lambda on the next iteration
        self.reuse_lambda = reuse_lambda

    def new_surrogate_obj(self, policy, input_var, Q_est_var, pi_old_vars, action_vars, lambda_var):
        probs_vars = policy.probs_vars
        N_var = input_var.shape[0]
        mean_kl = None
        lr = None
        for probs_var, pi_old_var, action_var in zip(probs_vars, pi_old_vars, action_vars):
            mean_kl = reduce_add(mean_kl, T.mean(T.sum(pi_old_var * (T.log(pi_old_var + 1e-6) - T.log(probs_var + 1e-6)), axis=1)))
            pi_old_selected = pi_old_var[T.arange(N_var), action_var]
            pi_selected = probs_var[T.arange(N_var), action_var]
            lr = reduce_mul(lr, pi_selected / (pi_old_selected + 1e-6))
        # formulate as a minimization problem
        surrogate_obj = - T.mean(lr * Q_est_var) + lambda_var * mean_kl
        return surrogate_obj, mean_kl

    def train(self, gen_policy, mdp):
        input_var = T.matrix('input') # N*Ds
        Q_est_var = T.vector('Q_est') # N
        pi_old_vars = [T.matrix('pi_old_%d' % i) for i in range(len(mdp.action_dims))] # (N*Da) * Na
        action_vars = [T.vector('action_%d' % i, dtype='uint8') for i in range(len(mdp.action_dims))] # (N) * Na
        lambda_var = T.scalar('lambda')

        tgt_policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)
        tgt_surrogate_obj, tgt_mean_kl = self.new_surrogate_obj(tgt_policy, input_var, Q_est_var, pi_old_vars, action_vars, lambda_var)
        # this is for the optimization
        opt_policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)
        opt_surrogate_obj, opt_mean_kl = self.new_surrogate_obj(opt_policy, input_var, Q_est_var, pi_old_vars, action_vars, lambda_var)

        opt_grads = theano.gradient.grad(opt_surrogate_obj, opt_policy.params)

        all_inputs = [input_var, Q_est_var] + pi_old_vars + action_vars + [lambda_var]

        with Message("Compiling functions..."):
            compute_opt_surrogate_obj = theano.function(all_inputs, opt_surrogate_obj, on_unused_input='ignore', allow_input_downcast=True)
            compute_tgt_mean_kl = theano.function(all_inputs, tgt_mean_kl, on_unused_input='ignore', allow_input_downcast=True)
            compute_opt_mean_kl = theano.function(all_inputs, opt_mean_kl, on_unused_input='ignore', allow_input_downcast=True)
            compute_opt_grads = theano.function(all_inputs, opt_grads, on_unused_input='ignore', allow_input_downcast=True)

        state, obs = mdp.sample_initial_state()

        for itr in xrange(self.n_itr):
            total_q_vals = defaultdict(int)
            action_visits = defaultdict(int)
            traj = []

            samples = []
            tot_rewards = 0
            n_traj = 0

            itr_log = prefix_log('itr #%d | ' % itr)

            for sample_itr in xrange(self.samples_per_itr):
                if (sample_itr + 1) % 1000 == 0:
                    itr_log('%d / %d samples' % (sample_itr + 1, self.samples_per_itr))
                actions, action_probs = tgt_policy.get_actions_single(obs)
                next_state, next_obs, reward, done = mdp.step_single(state, actions)
                traj.append((obs, actions, next_obs, reward))
                samples.append((obs, actions, action_probs))
                tot_rewards += reward
                if done or sample_itr == self.samples_per_itr - 1:
                    n_traj += 1
                    # update all Q-values along this trajectory
                    cum_reward = 0
                    for obs, actions, next_obs, reward in traj[::-1]:
                        cum_reward = self.discount * cum_reward + reward
                        action_pair = (tuple(obs), tuple(actions))
                        total_q_vals[action_pair] += cum_reward
                        action_visits[action_pair] += 1
                    traj = []
                state, obs = next_state, next_obs
            N = len(samples)
            all_obs = np.zeros((N,) + mdp.observation_shape)
            Q_est = np.zeros(N)
            all_pi_old = [np.zeros((N, Da)) for Da in mdp.action_dims]
            all_actions = [np.zeros(N, dtype='uint8') for _ in mdp.action_dims]
            for idx, tpl in enumerate(samples):
                obs, actions, action_probs = tpl
                for ia, action in enumerate(actions):
                    all_actions[ia][idx] = action
                for ia, probs in enumerate(action_probs):
                    all_pi_old[ia][idx,:] = probs
                action_pair = (tuple(obs), tuple(actions))
                Q_est[idx] = total_q_vals[action_pair] / action_visits[action_pair]
                all_obs[idx] = obs

            all_input_values = [all_obs, Q_est] + all_pi_old + all_actions

            def evaluate_cost(lambda_):
                def evaluate(params):
                    opt_policy.set_param_values(params)
                    val = compute_opt_surrogate_obj(*(all_input_values + [lambda_]))
                    return val
                return evaluate
            
            def evaluate_grad(lambda_):
                def evaluate(params):
                    opt_policy.set_param_values(params)
                    grad = compute_opt_grads(*(all_input_values + [lambda_]))
                    return flatten_tensors(map(np.asarray, grad))
                return evaluate

            cur_params = tgt_policy.get_param_values()
            itr_log('avg reward: %.3f over %d trajectories' % (tot_rewards * 1.0 / n_traj, n_traj))
            lambda_ = self.initial_lambda
            result = scipy.optimize.fmin_l_bfgs_b(func=evaluate_cost(lambda_), x0=cur_params, fprime=evaluate_grad(lambda_), maxiter=100)
            mean_kl = compute_opt_mean_kl(*(all_input_values + [lambda_]))
            itr_log('trying lambda=%.3f ... mean kl %.3f' % (lambda_, mean_kl))
            # do line search on lambda
            if mean_kl > self.stepsize:
                for _ in xrange(4):
                    lambda_ = lambda_ * 2
                    result = scipy.optimize.fmin_l_bfgs_b(func=evaluate_cost(lambda_), x0=cur_params, fprime=evaluate_grad(lambda_), maxiter=100)
                    mean_kl = compute_opt_mean_kl(*(all_input_values + [lambda_]))
                    if np.isnan(mean_kl): import ipdb; ipdb.set_trace()
                    itr_log('trying lambda=%.3f ... mean kl %.3f' % (lambda_, mean_kl))
                    if mean_kl <= self.stepsize:
                        break
            else:
                for _ in xrange(4):
                    try_lambda_ = lambda_ * 0.5
                    try_result = scipy.optimize.fmin_l_bfgs_b(func=evaluate_cost(try_lambda_), x0=cur_params, fprime=evaluate_grad(try_lambda_), maxiter=100)
                    try_mean_kl = compute_opt_mean_kl(*(all_input_values + [try_lambda_]))
                    itr_log('trying lambda=%.3f ... mean kl %.3f' % (try_lambda_, try_mean_kl))
                    if np.isnan(mean_kl): import ipdb; ipdb.set_trace()
                    if try_mean_kl > self.stepsize:
                        break
                    result = try_result
                    lambda_ = try_lambda_
                    mean_kl = try_mean_kl
                    
            result_x, result_f, result_d = result

            itr_log('optimization finished. new loss value: %.3f. mean kl: %.3f' % (result_f, mean_kl))
            sys.stdout.flush()
            
            tgt_policy.set_param_values(result_x)
