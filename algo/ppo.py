import numpy as np
import sys
from misc.console import SimpleMessage, prefix_log, tee_log, mkdir_p
from misc.tensor_utils import flatten_tensors
import misc.logger as logger
import multiprocessing
import cgtcompat as theano
import cgtcompat.tensor as T
import pydoc
from remote_sampler import RemoteSampler
import time
import itertools
import re
import scipy.optimize

def new_surrogate_obj(policy, input_var, Q_est_var, old_pdep_vars, action_var, penalty_var):
    pdep_vars = policy.pdep_vars
    kl = policy.kl(old_pdep_vars, pdep_vars)
    lr = policy.likelihood_ratio(old_pdep_vars, pdep_vars, action_var)
    mean_kl = T.mean(kl)
    # formulate as a minimization problem
    surrogate_loss = - T.mean(lr * Q_est_var)
    surrogate_obj = surrogate_loss + penalty_var * mean_kl
    return surrogate_obj, surrogate_loss, mean_kl

def get_train_vars(policy):
    input_var = policy.input_var
    Q_est_var = T.vector('Q_est') # N
    old_pdep_vars = [T.matrix('old_pdep_%d' % i) for i in range(len(policy.pdep_vars))]
    action_var = T.matrix('action')
    penalty_var = T.scalar('penalty')
    return dict(
        input_var=input_var,
        Q_est_var=Q_est_var,
        old_pdep_vars=old_pdep_vars,
        action_var=action_var,
        penalty_var=penalty_var,
    )

def merge_dict(x, y):
    z = x.copy()
    z.update(y)
    return z

def to_input_var_list(input_var, Q_est_var, old_pdep_vars, action_var, penalty_var):
    return [input_var, Q_est_var] + old_pdep_vars + [action_var, penalty_var]

def center_qval(qval):
    return (qval - np.mean(qval)) / (qval.std() + 1e-8)

# Proximal Policy Optimization
class PPO(object):

    def __init__(
            self, n_itr=500, start_itr=0, max_samples_per_itr=50000,
            discount=0.98, stepsize=0.015,
            initial_penalty=1, max_opt_itr=20, max_penalty_itr=10, exp_name='ppo',
            n_parallel=multiprocessing.cpu_count(), adapt_penalty=True,
            save_snapshot=True,
            optimizer=scipy.optimize.fmin_l_bfgs_b):
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.max_samples_per_itr = max_samples_per_itr
        self.discount = discount
        self.stepsize = stepsize
        self.initial_penalty = initial_penalty
        self.max_opt_itr = max_opt_itr
        self.max_penalty_itr = max_penalty_itr
        self.adapt_penalty = adapt_penalty
        self.save_snapshot = save_snapshot
        self.n_parallel = n_parallel
        self.exp_name = exp_name
        self.optimizer = optimizer

    def start_worker(self, gen_mdp, gen_policy):
        self.sampler = RemoteSampler('algo.rollout_sampler', self.n_parallel, gen_mdp, gen_policy)
        self.sampler.__enter__()

    def shutdown_worker(self):
        self.sampler.__exit__()

    # Main optimization loop
    def train(self, gen_mdp, gen_policy):
        logger.push_prefix('[%s] | ' % (self.exp_name))
        mdp = gen_mdp()
        policy = gen_policy(mdp)
        opt_info = self.init_opt(mdp, policy)
        self.start_worker(gen_mdp, gen_policy)
        for itr in xrange(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            samples_data = self.obtain_samples(itr, mdp, policy)
            opt_info = self.optimize_policy(itr, policy, samples_data, opt_info)
            self.perform_save_snapshot(itr, samples_data, opt_info)
            logger.pop_prefix()
        self.shutdown_worker()
        logger.pop_prefix()

    def init_opt(self, mdp, policy):
        train_vars = get_train_vars(policy)
        surr_obj, surr_loss, mean_kl = new_surrogate_obj(policy, **train_vars)
        grads = theano.gradient.grad(surr_obj, policy.params)
        input_list = to_input_var_list(**train_vars)
        logger.log("Compiling functions...")
        f_surr_kl = theano.function(
            input_list, [surr_obj, surr_loss, mean_kl], on_unused_input='ignore',
            allow_input_downcast=True
            )
        f_grads = theano.function(
            input_list, grads, on_unused_input='ignore',
            allow_input_downcast=True
            )
        penalty = self.initial_penalty
        return dict(
            f_surr_kl=f_surr_kl,
            f_grads=f_grads,
            penalty=penalty,
        )

    def obtain_samples(self, itr, mdp, policy):
        logger.log('collecting samples...')
        cur_params = policy.get_param_values()
        tot_rewards, n_traj, all_obs, Q_est, all_pdeps, all_actions = \
            self.sampler.request_samples(
                itr, cur_params, self.max_samples_per_itr,
                self.discount)
        Q_est = center_qval(Q_est)
        all_input_values = [all_obs, Q_est] + all_pdeps + [all_actions]

        avg_reward = tot_rewards * 1.0 / n_traj

        ent = policy.compute_entropy(all_pdeps)

        logger.log('entropy: %f' % ent)
        logger.log('perplexity: %f' % np.exp(ent))
        logger.log('avg reward: %f over %d trajectories' % (avg_reward, n_traj))

        return dict(
            all_input_values=all_input_values,
            all_obs=all_obs,
            Q_est=Q_est,
            all_actions=all_actions,
            all_pdeps=all_pdeps,
        )

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        penalty = opt_info['penalty']
        f_surr_kl = opt_info['f_surr_kl']
        f_grads = opt_info['f_grads']
        all_input_values = samples_data['all_input_values']
        cur_params = policy.get_param_values()

        def evaluate_cost(penalty):
            def evaluate(params):
                policy.set_param_values(params)
                inputs_with_penalty = all_input_values + [penalty]
                val, loss, mean_kl = f_surr_kl(*inputs_with_penalty)
                return val.astype(np.float64)
            return evaluate

        def evaluate_grad(penalty):
            def evaluate(params):
                policy.set_param_values(params)
                grad = f_grads(*(all_input_values + [penalty]))
                flattened_grad = flatten_tensors(map(np.asarray, grad))
                return flattened_grad.astype(np.float64)
            return evaluate

        loss_before = evaluate_cost(0)(cur_params)
        logger.log('loss before: %f' % loss_before)

        penalty = np.clip(penalty, 1e-2, 1e6)

        # search for the best penalty parameter
        penalty_scale_factor = None
        opt_params = None
        max_penalty_itr = self.max_penalty_itr
        for penalty_itr in range(max_penalty_itr):
            logger.log('trying penalty=%.3f...' % penalty)
            result = self.optimizer(
                func=evaluate_cost(penalty), x0=cur_params,
                fprime=evaluate_grad(penalty),
                maxiter=self.max_opt_itr
                )
            loss, _, mean_kl = f_surr_kl(*(all_input_values + [penalty]))
            logger.log('penalty %f => loss %f, mean kl %f' % (penalty, loss, mean_kl))
            if mean_kl < self.stepsize:
                opt_params = policy.get_param_values()
                final_penalty = penalty

            if not self.adapt_penalty:
                break

            # decide scale factor on the first iteration
            if penalty_scale_factor is None:
                if mean_kl > self.stepsize:
                    # need to increase penalty
                    penalty_scale_factor = 2
                else:
                    # can shrink penalty
                    penalty_scale_factor = 0.5
            else:
                if penalty_scale_factor > 1 and mean_kl <= self.stepsize:
                    break
                elif penalty_scale_factor < 1 and mean_kl >= self.stepsize:
                    break
            penalty *= penalty_scale_factor

        if opt_params is None:
            opt_params = policy.get_param_values()
            final_penalty = penalty

        loss_after = evaluate_cost(0)(opt_params)
        logger.log('optimization finished. loss after: %f. mean kl: %f. dloss: %f' % (loss_after, mean_kl, loss_before - loss_after))
        policy.set_param_values(opt_params)

        return merge_dict(opt_info, dict(
            cur_params=cur_params,
            opt_params=opt_params,
            penalty=final_penalty,
        ))

    def perform_save_snapshot(self, itr, samples_data, opt_info):
        if self.save_snapshot:
            logger.log("saving result...")
            savedir = 'data/%s' % (self.exp_name)
            mkdir_p(savedir)
            to_save = {
                'itr': itr,
                'cur_policy_params': opt_info['cur_params'],
                'opt_policy_params': opt_info['opt_params'],
                'all_obs': samples_data['all_obs'],
                'Q_est': samples_data['Q_est'],
                'penalty': opt_info['penalty'],
                'actions': samples_data['all_actions'],
            }
            for idx, pdep in enumerate(samples_data['all_pdeps']):
                to_save['pdep_%d' % idx] = pdep
            np.savez_compressed('%s/itr_%03d.npz' % (savedir, itr), **to_save)
