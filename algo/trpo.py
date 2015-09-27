import os
import numpy as np
import scipy.optimize
import lasagne.layers as L
import operator
import sys
from misc.logging import Message, log, prefix_log
from misc.tensor_utils import flatten_tensors, unflatten_tensors
from collections import defaultdict
import multiprocessing
from joblib.pool import MemmapingPool
from joblib.parallel import SafeFunction
import theano
import theano.tensor as T
import theano.sandbox.cuda


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

def collect_samples(args):
    try:
        param_values, n_samples, discount = args
        global policy
        global mdp
        total_q_vals = defaultdict(int)
        action_visits = defaultdict(int)
        traj = []
        samples = []
        tot_rewards = 0
        n_traj = 0

        log('starting...')
        policy.set_param_values(param_values)

        effective_steps = 0
        last_displayed = 0

        state, obs = mdp.sample_initial_state()

        while effective_steps < n_samples:
            if effective_steps / 1000 > last_displayed:
                log('%d / %d steps (%d samples; %d traj)' % (effective_steps, n_samples, len(samples), n_traj))
                last_displayed += 1
            actions, action_probs = policy.get_actions_single(obs)
            next_state, next_obs, reward, done, effective_step = mdp.step_single(state, actions)
            effective_steps += effective_step
            traj.append((obs, actions, next_obs, reward))
            samples.append((obs, actions, action_probs))
            tot_rewards += reward
            if done or effective_steps >= n_samples:
                n_traj += 1
                # update all Q-values along this trajectory
                cum_reward = 0
                for obs, actions, next_obs, reward in traj[::-1]:
                    cum_reward = discount * cum_reward + reward
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

        return tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions
    except Exception as e:
        import traceback
        traceback.print_exc(e)
        raise e

def combine_samples(results):
    rewards_list, n_traj_list, all_obs_list, Q_est_list, all_pi_old_list, all_actions_list = map(list, zip(*results))
    tot_rewards = sum(rewards_list)
    n_traj = sum(n_traj_list)
    all_obs = np.concatenate(all_obs_list)
    Q_est = np.concatenate(Q_est_list)
    na = len(all_pi_old_list[0])
    all_pi_old = [np.concatenate(map(lambda x: x[i], all_pi_old_list)) for i in range(na)]
    all_actions = [np.concatenate(map(lambda x: x[i], all_actions_list)) for i in range(na)]
    return tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions

def initialize_collector_state(gen_policy, gen_mdp):
    global policy
    global mdp
    global state
    global obs

    np.random.seed(os.getpid())

    mdp = gen_mdp()
    input_var = T.matrix('input') # N*Ds
    policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)
    pid = np.random.rand()

class TRPO(object):

    def __init__(self, n_itr=500, samples_per_itr=100000, discount=0.99,
            stepsize=0.015, initial_lambda=1, max_itr=100, n_parallel=multiprocessing.cpu_count(),
            adapt_lambda=True, reuse_lambda=True):
        self.n_itr = n_itr
        self.samples_per_itr = samples_per_itr
        self.discount = discount
        self.stepsize = stepsize
        self.initial_lambda = initial_lambda
        self.max_itr = max_itr
        self.adapt_lambda = adapt_lambda
        self.n_parallel = n_parallel
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


    def train(self, gen_policy, gen_mdp):

        mdp = gen_mdp()
        input_var = T.matrix('input') # N*Ds
        Q_est_var = T.vector('Q_est') # N
        pi_old_vars = [T.matrix('pi_old_%d' % i) for i in range(len(mdp.action_dims))] # (N*Da) * Na
        action_vars = [T.vector('action_%d' % i, dtype='uint8') for i in range(len(mdp.action_dims))] # (N) * Na
        lambda_var = T.scalar('lambda')

        # this is for the optimization
        policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)
        surrogate_obj, mean_kl = self.new_surrogate_obj(policy, input_var, Q_est_var, pi_old_vars, action_vars, lambda_var)

        grads = theano.gradient.grad(surrogate_obj, policy.params)

        all_inputs = [input_var, Q_est_var] + pi_old_vars + action_vars + [lambda_var]

        with Message("Compiling functions..."):
            compute_surrogate_obj = theano.function(all_inputs, surrogate_obj, on_unused_input='ignore', allow_input_downcast=True)
            compute_mean_kl = theano.function(all_inputs, mean_kl, on_unused_input='ignore', allow_input_downcast=True)
            compute_grads = theano.function(all_inputs, grads, on_unused_input='ignore', allow_input_downcast=True)

        lambda_ = self.initial_lambda

        theano.sandbox.cuda.unuse()

        pool = MemmapingPool(self.n_parallel, initializer=initialize_collector_state, initargs=(gen_policy, gen_mdp))

        #initialize_collector_state(gen_policy, gen_mdp)
        for itr in xrange(self.n_itr):
            total_q_vals = defaultdict(int)
            action_visits = defaultdict(int)

            itr_log = prefix_log('itr #%d | ' % (itr + 1))

            args = (policy.get_param_values(), self.samples_per_itr / self.n_parallel, self.discount)
            #collect_samples(args)
            result_list = pool.map(SafeFunction(collect_samples), [args] * self.n_parallel)
            tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions = \
                    combine_samples(result_list)
                            
            theano.sandbox.cuda.use('gpu0', force=True)
            all_input_values = [all_obs, Q_est] + all_pi_old + all_actions

            def evaluate_cost(lambda_):
                def evaluate(params):
                    policy.set_param_values(params)
                    val = compute_surrogate_obj(*(all_input_values + [lambda_]))
                    return val.astype(np.float64)
                return evaluate
            
            def evaluate_grad(lambda_):
                def evaluate(params):
                    policy.set_param_values(params)
                    grad = compute_grads(*(all_input_values + [lambda_]))
                    return flatten_tensors(map(np.asarray, grad)).astype(np.float64)
                return evaluate

            # what if we reset the policy parameters every iteration?
            #reset_policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)
            #policy.set_param_values(reset_policy.get_param_values())
            cur_params = policy.get_param_values()
            itr_log('avg reward: %.3f over %d trajectories' % (tot_rewards * 1.0 / n_traj, n_traj))
            if not self.reuse_lambda:
                lambda_ = self.initial_lambda
            result = scipy.optimize.fmin_l_bfgs_b(func=evaluate_cost(lambda_), x0=cur_params, fprime=evaluate_grad(lambda_), maxiter=self.max_itr)
            mean_kl = compute_mean_kl(*(all_input_values + [lambda_]))
            itr_log('trying lambda=%.3f ... mean kl %.3f' % (lambda_, mean_kl))
            # do line search on lambda
            if self.adapt_lambda:
                max_search = 4
                if itr < 2:
                    max_search = 10
                if mean_kl > self.stepsize:
                    for _ in xrange(max_search):
                        lambda_ = lambda_ * 2
                        result = scipy.optimize.fmin_l_bfgs_b(func=evaluate_cost(lambda_), x0=cur_params, fprime=evaluate_grad(lambda_), maxiter=self.max_itr)
                        mean_kl = compute_mean_kl(*(all_input_values + [lambda_]))
                        if np.isnan(mean_kl): import ipdb; ipdb.set_trace()
                        itr_log('trying lambda=%.3f ... mean kl %.3f' % (lambda_, mean_kl))
                        if mean_kl <= self.stepsize:
                            break
                else:
                    for _ in xrange(max_search):
                        try_lambda_ = lambda_ * 0.5
                        try_result = scipy.optimize.fmin_l_bfgs_b(func=evaluate_cost(try_lambda_), x0=cur_params, fprime=evaluate_grad(try_lambda_), maxiter=self.max_itr)
                        try_mean_kl = compute_mean_kl(*(all_input_values + [try_lambda_]))
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

            theano.sandbox.cuda.unuse()
