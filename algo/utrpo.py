import numpy as np
import sys
from misc.console import SimpleMessage, prefix_log, tee_log, mkdir_p
from misc.tensor_utils import flatten_tensors
import multiprocessing
import theano
import theano.tensor as T
import pydoc
from remote_sampler import RemoteSampler
import time
import itertools
import re
from lbfgs import lbfgs

# Unconstrained TRPO
class UTRPO(object):

    def __init__(
            self, n_itr=500, max_samples_per_itr=100000,
            max_steps_per_itr=np.inf, discount=0.99, stepsize=0.015,
            initial_lambda=1, max_opt_itr=100, exp_name='utrpo',
            n_parallel=multiprocessing.cpu_count(), adapt_lambda=True,
            reuse_lambda=True, sampler_module='algo.rollout_sampler',
            resume_file=None, optimizer_module='scipy.optimize.fmin_l_bfgs_b'):
        self.n_itr = n_itr
        self.max_samples_per_itr = max_samples_per_itr
        self.max_steps_per_itr = max_steps_per_itr
        self.discount = discount
        self.stepsize = stepsize
        self.initial_lambda = initial_lambda
        self.max_opt_itr = max_opt_itr
        self.adapt_lambda = adapt_lambda
        self.n_parallel = n_parallel
        self.exp_name = exp_name
        # whether to start from the currently adapted lambda on the next
        # iteration
        self.reuse_lambda = reuse_lambda
        self.sampler_module = sampler_module
        self.optimizer_module = optimizer_module
        self.resume_file = None

    def new_surrogate_obj(
            self, policy, input_var, Q_est_var, pi_old_vars, action_vars,
            lambda_var):
        probs_vars = policy.probs_vars
        N_var = input_var.shape[0]
        kl = 0
        lr = 1
        for probs_var, pi_old_var, action_var in zip(
                probs_vars, pi_old_vars, action_vars):
            kl += T.sum(
                    pi_old_var * (
                        T.log(pi_old_var) - T.log(probs_var)
                    ),
                    axis=1
                )
            pi_old_selected = pi_old_var[T.arange(N_var), action_var]
            pi_selected = probs_var[T.arange(N_var), action_var]
            lr *= pi_selected / (pi_old_selected)
        mean_kl = T.mean(kl)
        max_kl = T.max(kl)
        # formulate as a minimization problem
        surrogate_loss = - T.mean(lr * Q_est_var)
        surrogate_obj = surrogate_loss + lambda_var * mean_kl
        return surrogate_obj, surrogate_loss, mean_kl, max_kl

    def transform_gen_mdp(self, gen_mdp):
        return gen_mdp

    def transform_gen_policy(self, gen_policy):
        return gen_policy

    def train(self, gen_mdp, gen_policy):

        gen_mdp = self.transform_gen_mdp(gen_mdp)
        gen_policy = self.transform_gen_policy(gen_policy)

        exp_timestamp = time.strftime("%Y%m%d%H%M%S")
        mdp = gen_mdp()
        input_var = T.matrix('input')  # N*Ds
        policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)

        Q_est_var = T.vector('Q_est')  # N
        action_range = range(len(policy.action_dims))
        pi_old_vars = [T.matrix('pi_old_%d' % i) for i in action_range]
        action_vars = [T.vector('action_%d' % i, dtype='uint8')
                       for i in action_range]
        lambda_var = T.scalar('lambda')

        surrogate_obj, surrogate_loss, mean_kl, max_kl = \
            self.new_surrogate_obj(
                policy, input_var, Q_est_var, pi_old_vars, action_vars,
                lambda_var)

        grads = theano.gradient.grad(surrogate_obj, policy.params)

        all_inputs = [input_var, Q_est_var] + pi_old_vars + action_vars + \
            [lambda_var]

        exp_logger = prefix_log('[%s] | ' % (self.exp_name))

        with SimpleMessage("Compiling functions...", exp_logger):
            compute_surrogate_kl = theano.function(
                all_inputs, [surrogate_obj, mean_kl], on_unused_input='ignore',
                allow_input_downcast=True
                )
            compute_mean_kl = theano.function(
                all_inputs, mean_kl, on_unused_input='ignore',
                allow_input_downcast=True
                )
            compute_max_kl = theano.function(
                all_inputs, max_kl, on_unused_input='ignore',
                allow_input_downcast=True
                )

            compute_grads = theano.function(
                all_inputs, grads, on_unused_input='ignore',
                allow_input_downcast=True
                )

        optimizer = pydoc.locate(self.optimizer_module)

        logger = tee_log(self.exp_name + '_' + exp_timestamp + '.log')

        savedir = 'data/%s_%s' % (self.exp_name, exp_timestamp)
        mkdir_p(savedir)

        lambda_ = self.initial_lambda

        with RemoteSampler(
                self.sampler_module, self.n_parallel, gen_mdp,
                gen_policy, savedir) as sampler:

            if self.resume_file is not None:
                print 'Resuming from snapshot %s...' % self.resume_file
                resume_data = np.load(self.resume_file)
                start_itr = int(re.search('itr_(\d+)', self.resume_file).group(1)) + 1
                policy.set_param_values(resume_data['opt_policy_params'])
            else:
                start_itr = 0


            for itr in xrange(start_itr, self.n_itr):

                itr_log = prefix_log('[%s] itr #%d | ' % (self.exp_name, itr), logger=logger)

                cur_params = policy.get_param_values()

                itr_log('collecting samples...')

                tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions = \
                    sampler.request_samples(
                        itr, cur_params, self.max_samples_per_itr,
                        self.max_steps_per_itr, self.discount)

                Q_est = Q_est - np.mean(Q_est)
                Q_est = Q_est / (Q_est.std() + 1e-8)

                all_input_values = [all_obs, Q_est] + all_pi_old + all_actions

                def evaluate_cost(lambda_):
                    def evaluate(params):
                        policy.set_param_values(params)
                        inputs_with_lambda = all_input_values + [lambda_]
                        val, mean_kl = compute_surrogate_kl(*inputs_with_lambda)
                        if mean_kl > self.stepsize:
                            return np.inf
                        else:
                            return val.astype(np.float64)
                        #return val.astype(np.float64)
                    return evaluate

                def evaluate_grad(lambda_):
                    def evaluate(params):
                        policy.set_param_values(params)
                        grad = compute_grads(*(all_input_values + [lambda_]))
                        flattened_grad = flatten_tensors(map(np.asarray, grad))
                        #print 'grad norm: ', np.linalg.norm(flattened_grad)
                        return flattened_grad.astype(np.float64)
                    return evaluate

                avg_reward = tot_rewards * 1.0 / n_traj

                ent = 0
                for pi_old in all_pi_old:
                    ent += np.mean(np.sum(-pi_old * np.log(pi_old), axis=1))
                itr_log('entropy: %f' % ent)
                itr_log('perplexity: %f' % np.exp(ent))


                itr_log('avg reward: %f over %d trajectories' %
                        (avg_reward, n_traj))

                loss_before = evaluate_cost(0)(cur_params)
                itr_log('loss before: %f' % loss_before)

                if not self.reuse_lambda:
                    lambda_ = self.initial_lambda
                else:
                    lambda_ = min(10000, max(0.01, lambda_))

                with SimpleMessage('trying lambda=%.3f...' % lambda_, itr_log):
                    #opt_val = None
                    
                    #for n_itr, opt_val in enumerate(lbfgs(f=evaluate_cost(lambda_), fgrad=evaluate_grad(lambda_), x0=cur_params, maxiter=20)):
                    #    pass
                    ##itr_log('took %d itr' % n_itr)
                    #policy.set_param_values(opt_val)
                    result = optimizer(
                        func=evaluate_cost(lambda_), x0=cur_params,
                        fprime=evaluate_grad(lambda_),
                        maxiter=self.max_opt_itr
                        )
                    
                    mean_kl = compute_mean_kl(*(all_input_values + [lambda_]))
                    itr_log('lambda %f => mean kl %f' % (lambda_, mean_kl))
                # do line search on lambda
                if self.adapt_lambda:
                    max_search = 10
                    if itr - start_itr < 2:
                        max_search = 10
                    if mean_kl > self.stepsize:
                        for _ in xrange(max_search):
                            lambda_ = lambda_ * 2
                            with SimpleMessage('trying lambda=%f...' % lambda_, itr_log):
                                result = optimizer(
                                    func=evaluate_cost(lambda_), x0=cur_params,
                                    fprime=evaluate_grad(lambda_),
                                    maxiter=self.max_opt_itr)
                                inputs_with_lambda = all_input_values + [lambda_]
                                mean_kl = compute_mean_kl(*inputs_with_lambda)
                                itr_log('lambda %f => mean kl %f' % (lambda_, mean_kl))
                            if np.isnan(mean_kl):
                                import ipdb
                                ipdb.set_trace()
                            if mean_kl <= self.stepsize:
                                break
                    else:
                        for _ in xrange(max_search):
                            try_lambda_ = lambda_ * 0.5
                            with SimpleMessage('trying lambda=%f...' % try_lambda_, itr_log):
                                try_result = optimizer(
                                    func=evaluate_cost(try_lambda_), x0=cur_params,
                                    fprime=evaluate_grad(try_lambda_),
                                    maxiter=self.max_opt_itr)
                                inputs_with_lambda = all_input_values + [lambda_]
                                try_mean_kl = compute_mean_kl(*inputs_with_lambda)
                                itr_log('lambda=%f => mean kl %f' % (try_lambda_, try_mean_kl))
                            if np.isnan(mean_kl):
                                import ipdb
                                ipdb.set_trace()
                            if try_mean_kl > self.stepsize:
                                break
                            result = try_result
                            lambda_ = try_lambda_
                            mean_kl = try_mean_kl

                loss_after = evaluate_cost(0)(policy.get_param_values())
                itr_log('optimization finished. loss after: %f. mean kl: %f. dloss: %f' % (loss_after, mean_kl, loss_before - loss_after))
                timestamp = time.strftime("%Y%m%d%H%M%S")
                #result_x, result_f, result_d = result
                #itr_log('optimization finished. new loss value: %.3f. mean kl: %.3f' % (result_f, mean_kl))
                with SimpleMessage("saving result...", exp_logger):
                    to_save = {
                        'cur_policy_params': cur_params,
                        'opt_policy_params': policy.get_param_values(),
                        'all_obs': all_obs,
                        'Q_est': Q_est,
                        'itr': itr,
                        'lambda': lambda_,
                        'loss': loss_after,
                        'mean_kl': mean_kl,
                    }
                    for idx, pi_old, actions in zip(itertools.count(), all_pi_old, all_actions):
                        to_save['pi_old_%d' % idx] = pi_old
                        to_save['actions_%d' % idx] = actions
                    np.savez_compressed('%s/itr_%d_%s.npz' % (savedir, itr, timestamp), **to_save)
                sys.stdout.flush()
