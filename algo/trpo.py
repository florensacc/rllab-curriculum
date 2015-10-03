import numpy as np
import sys
from misc.console import SimpleMessage, prefix_log, tee_log
from misc.tensor_utils import flatten_tensors, unflatten_tensors, cg
import multiprocessing
import theano
import theano.tensor as T
import pydoc
from remote_sampler import RemoteSampler
import time
import itertools

# Unconstrained TRPO
class TRPO(object):

    def __init__(
            self, n_itr=500, max_samples_per_itr=100000,
            max_steps_per_itr=np.inf, discount=0.99, stepsize=0.015,
            max_opt_itr=10, exp_name='trpo',
            n_parallel=multiprocessing.cpu_count(), adapt_lambda=True,
            reuse_lambda=True, sampler_module='algo.rollout_sampler'):
        self._n_itr = n_itr
        self._max_samples_per_itr = max_samples_per_itr
        self._max_steps_per_itr = max_steps_per_itr
        self._discount = discount
        self._stepsize = stepsize
        self._max_opt_itr = max_opt_itr
        self._adapt_lambda = adapt_lambda
        self._n_parallel = n_parallel
        self._exp_name = exp_name
        # whether to start from the currently adapted lambda on the next
        # iteration
        self._reuse_lambda = reuse_lambda
        self._sampler_module = sampler_module

    def new_surrogate_obj(
            self, policy, input_var, Q_est_var, pi_old_vars, action_vars
        ):
        probs_vars = policy.probs_vars
        N_var = input_var.shape[0]
        mean_kl = 0
        lr = 1
        for probs_var, pi_old_var, action_var in zip(
                probs_vars, pi_old_vars, action_vars):
            mean_kl += T.mean(
                T.sum(
                    pi_old_var * (
                        T.log(pi_old_var + 1e-6) - T.log(probs_var + 1e-6)
                    ),
                    axis=1
                ))
            pi_old_selected = pi_old_var[T.arange(N_var), action_var]
            pi_selected = probs_var[T.arange(N_var), action_var]
            lr *= pi_selected / (pi_old_selected + 1e-6)
        # formulate as a minimization problem
        surrogate_loss = - T.mean(lr * Q_est_var)
        return surrogate_loss, mean_kl

    def transform_gen_mdp(self, gen_mdp):
        return gen_mdp

    def transform_gen_policy(self, gen_policy):
        return gen_policy

    def train(self, gen_mdp, gen_policy):

        gen_mdp = self.transform_gen_mdp(gen_mdp)
        gen_policy = self.transform_gen_policy(gen_policy)

        mdp = gen_mdp()
        input_var = T.matrix('input')  # N*Ds
        policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)

        Q_est_var = T.vector('Q_est')  # N
        action_range = range(len(policy.action_dims))
        pi_old_vars = [T.matrix('pi_old_%d' % i) for i in action_range]
        action_vars = [T.vector('action_%d' % i, dtype='uint8')
                       for i in action_range]

        N_var = input_var.shape[0]

        surrogate_loss, mean_kl = self.new_surrogate_obj(
                policy, input_var, Q_est_var, pi_old_vars, action_vars)


        #flattened_params = T.concatenate(map(T.flatten, policy.params))

        grads = T.grad(surrogate_loss, policy.params)
        
        # This is kind of annoying, but this Fisher-vector product function
        # takes sliced vectors as input, and outputs flattened products
        ys = map(lambda x: x.type(x.name + '_input'), policy.params)#T.vector('y')
        jtmjy = 0

        for probs_var, pi_old_var in zip(policy.probs_vars, pi_old_vars):
            m = T.grad(mean_kl, probs_var)
            jy = T.Rop(probs_var, policy.params, ys)
            mjy = policy.fisher_vector_product(pi_old_var, probs_var, jy)
            jtmjy_raw = T.Lop(probs_var, policy.params, mjy)
            jtmjy += T.concatenate(map(T.flatten, jtmjy_raw)) / N_var


        all_inputs = [input_var, Q_est_var] + pi_old_vars + action_vars

        with SimpleMessage("Compiling functions..."):
            compute_surrogate_loss = theano.function(
                all_inputs, surrogate_loss, on_unused_input='ignore',
                allow_input_downcast=True
                )
            compute_grads = theano.function(
                all_inputs, grads, on_unused_input='ignore',
                allow_input_downcast=True
                )
            compute_jtmjy = theano.function(all_inputs + ys, jtmjy, on_unused_input='ignore', allow_input_downcast=True)


        logger = tee_log(self._exp_name + '.log')

        with RemoteSampler(
                self._sampler_module, self._n_parallel, gen_mdp,
                gen_policy) as sampler:

            for itr in xrange(self._n_itr):

                itr_log = prefix_log('itr #%d | ' % (itr + 1), logger=logger)

                cur_params = policy.get_param_values()

                itr_log('collecting samples...')

                tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions = \
                    sampler.request_samples(
                        itr, cur_params, self._max_samples_per_itr,
                        self._max_steps_per_itr, self._discount)

                all_input_values = [all_obs, Q_est] + all_pi_old + all_actions

                grads_computed = flatten_tensors(compute_grads(*all_input_values))

                def evaluate_fisher_prod(params):
                    unflattened = unflatten_tensors(params, policy.param_shapes)
                    return compute_jtmjy(*(all_input_values + unflattened))

                #evaluate_fisher_prod(cur_params)
                with SimpleMessage('computing direction...', itr_log):
                    direction = cg(evaluate_fisher_prod, grads_computed, np.zeros(grads_computed.shape), max_itr=10)

                beta = (2 * self._stepsize / np.dot(direction.T, evaluate_fisher_prod(direction))) ** 0.5

                def evaluate_cost(params):
                    policy.set_param_values(params)
                    val = compute_surrogate_loss(*all_input_values)
                    return float(val.astype(np.float64))

                def evaluate_grad(params):
                    policy.set_param_values(params)
                    grad = compute_grads(*all_input_values)
                    flattened_grad = flatten_tensors(map(np.asarray, grad))
                    return flattened_grad.astype(np.float64)

                import ipdb; ipdb.set_trace()

                avg_reward = tot_rewards * 1.0 / n_traj

                itr_log('avg reward: %.3f over %d trajectories' %
                        (avg_reward, n_traj))


                gval = evaluate_grad(cur_params)

                # First, we need to compute a search direction, by approximately
                # solving the equation Ax = -g, where A is the Fisher information
                # matrix
                 


                with SimpleMessage('trying lambda=%.3f...' % lambda_, itr_log):
                    result = optimizer(
                        func=evaluate_cost(lambda_), x0=cur_params,
                        fprime=evaluate_grad(lambda_),
                        maxiter=self._max_opt_itr
                        )
                    mean_kl = compute_mean_kl(*(all_input_values + [lambda_]))
                    itr_log('lambda %.3f => mean kl %.3f' % (lambda_, mean_kl))

                timestamp = time.strftime("%Y%m%d%H%M%S")
                result_x, result_f, result_d = result
                itr_log('optimization finished. new loss value: %.3f. mean kl: %.3f' % (result_f, mean_kl))
                itr_log('saving result...')
                to_save = {
                    'cur_policy_params': cur_params,
                    'opt_policy_params': policy.get_param_values(),
                    'all_obs': all_obs,
                    'Q_est': Q_est,
                }
                for idx, pi_old, actions in zip(itertools.count(), all_pi_old, all_actions):
                    to_save['pi_old_%d' % idx] = pi_old
                    to_save['actions_%d' % idx] = actions
                np.savez_compressed('data/%s_itr_%d_%s.npz' % (self._exp_name, itr, timestamp), **to_save)
                sys.stdout.flush()
