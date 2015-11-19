from rllab.misc.console import mkdir_p
from rllab.misc.tensor_utils import flatten_tensors
from rllab.misc.special import discount_cumsum, explained_variance_1d
from rllab.sampler import parallel_sampler
from rllab.misc.ext import merge_dict
from rllab.misc import autoargs
from rllab.algo.base import Algorithm
import rllab.plotter as plotter
import rllab.misc.logger as logger
import multiprocessing
import tensorfuse as theano
import tensorfuse.tensor as T
import scipy.optimize
import numpy as np


def new_surrogate_obj(policy, input_var, Q_est_var, old_pdist_var, action_var, penalty_var):
    pdist_var = policy.pdist_var
    kl = policy.kl(old_pdist_var, pdist_var)
    lr = policy.likelihood_ratio(old_pdist_var, pdist_var, action_var)
    mean_kl = T.mean(kl)
    # formulate as a minimization problem
    surrogate_loss = - T.mean(lr * Q_est_var)
    surrogate_obj = surrogate_loss + penalty_var * mean_kl
    return surrogate_obj, surrogate_loss, mean_kl


def get_train_vars(policy):
    input_var = policy.input_var
    Q_est_var = T.vector('Q_est') # N
    old_pdist_var = T.matrix('old_pdist')
    action_var = policy.new_action_var('action')
    penalty_var = T.scalar('penalty')
    return dict(
        input_var=input_var,
        Q_est_var=Q_est_var,
        old_pdist_var=old_pdist_var,
        action_var=action_var,
        penalty_var=penalty_var,
    )


def to_input_var_list(input_var, Q_est_var, old_pdist_var, action_var, penalty_var):
    return [input_var, Q_est_var, old_pdist_var, action_var, penalty_var]


def center_qval(qval):
    return (qval - np.mean(qval)) / (qval.std() + 1e-8)


# Proximal Policy Optimization
class PPO(Algorithm):

    @autoargs.arg("n_itr", type=str, help="Number of iterations.")
    @autoargs.arg("samples_per_itr", type=int, help="Number of samples per iteration.")
    @autoargs.arg("max_path_length", type=int, help="Maximum length of a single rollout.")
    @autoargs.arg("discount", type=int, help="Discount.")
    @autoargs.arg("gae_lambda", type=int, help="Lambda used for generalized advantage estimation.")
    @autoargs.arg("stepsize", type=int, help="Maximum change in mean KL per iteration.")
    def __init__(
            self,
            n_itr=500,
            start_itr=0,
            samples_per_itr=50000,
            max_path_length=np.inf,
            discount=0.98,
            gae_lambda=1,
            stepsize=0.015,
            initial_penalty=1,
            max_opt_itr=20,
            max_penalty_itr=10,
            exp_name='ppo',
            adapt_penalty=True,
            n_parallel=multiprocessing.cpu_count(),
            save_snapshot=True,
            optimizer=scipy.optimize.fmin_l_bfgs_b,
            plot=False):
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.samples_per_itr = samples_per_itr
        self.max_path_length = max_path_length
        self.discount = discount
        self.stepsize = stepsize
        self.initial_penalty = initial_penalty
        self.max_opt_itr = max_opt_itr
        self.max_penalty_itr = max_penalty_itr
        self.exp_name = exp_name
        self.adapt_penalty = adapt_penalty
        self.n_parallel = n_parallel
        self.save_snapshot = save_snapshot
        self.optimizer = optimizer
        self.gae_lambda = gae_lambda
        self.plot = plot

    def start_worker(self, mdp, policy, vf):
        parallel_sampler.populate_task(mdp, policy)
        if self.plot:
            plotter.init_plot(mdp, policy)

    def shutdown_worker(self):
        pass

    def update_plot(self, policy):
        if self.plot:
            plotter.update_plot(policy, self.max_path_length)

    # Main optimization loop
    def train(self, mdp, policy, vf):
        savedir = 'data/%s' % (self.exp_name)
        logger.add_file_output(savedir + '/log.txt')
        logger.push_prefix('[%s] | ' % (self.exp_name))
        opt_info = self.init_opt(mdp, policy, vf)
        self.start_worker(mdp, policy, vf)
        for itr in xrange(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            samples_data = self.obtain_samples(itr, mdp, policy, vf)
            opt_info = self.optimize_policy(itr, policy, samples_data, opt_info)
            self.perform_save_snapshot(itr, samples_data, opt_info)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                self.update_plot(policy)
        self.shutdown_worker()
        logger.remove_file_output(savedir + '/log.txt')
        logger.pop_prefix()

    def init_opt(self, mdp, policy, vf):
        train_vars = get_train_vars(policy)
        surr_obj, surr_loss, mean_kl = new_surrogate_obj(policy, **train_vars)
        grads = theano.gradient.grad(surr_obj, policy.params)
        input_list = to_input_var_list(**train_vars)
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

    def obtain_samples(self, itr, mdp, policy, vf):
        cur_params = policy.get_param_values()
        paths = parallel_sampler.request_samples(cur_params, self.samples_per_itr, self.max_path_length)

        all_baselines = []
        all_returns = []

        for path in paths:
            path["returns"] = discount_cumsum(path["rewards"], self.discount)
            baselines = np.append(vf.predict(path), 0)
            deltas = path["rewards"] + self.discount*baselines[1:] - baselines[:-1]
            path["advantage"] = discount_cumsum(deltas, self.discount*self.gae_lambda)
            all_baselines.append(baselines[:-1])
            all_returns.append(path["returns"])

        ev = explained_variance_1d(np.concatenate(all_baselines), np.concatenate(all_returns))

        all_obs = np.vstack([path["observations"] for path in paths])
        all_states = np.vstack([path["states"] for path in paths])
        all_pdists = np.vstack([path["pdists"] for path in paths])
        all_actions = np.vstack([path["actions"] for path in paths])
        all_advantages = np.concatenate([path["advantage"] for path in paths])

        avg_return = np.mean([sum(path["rewards"]) for path in paths])

        Q_est = center_qval(all_advantages)

        all_input_values = [all_obs, Q_est, all_pdists, all_actions]

        ent = policy.compute_entropy(all_pdists)

        # Update vf
        vf.fit(paths)

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('AvgReturn', avg_return)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('ExplainedVariance', ev)

        return dict(
            all_input_values=all_input_values,
            all_obs=all_obs,
            Q_est=Q_est,
            all_actions=all_actions,
            all_pdists=all_pdists,
            paths=paths,
            vf_params=vf.get_param_values(),
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
        logger.record_tabular('LossBefore', loss_before)

        try_penalty = np.clip(penalty, 1e-2, 1e6)

        # search for the best penalty parameter
        penalty_scale_factor = None
        opt_params = None
        max_penalty_itr = self.max_penalty_itr
        mean_kl = None
        for penalty_itr in range(max_penalty_itr):
            logger.log('trying penalty=%.3f...' % try_penalty)
            result = self.optimizer(
                func=evaluate_cost(try_penalty), x0=cur_params,
                fprime=evaluate_grad(try_penalty),
                maxiter=self.max_opt_itr
                )
            _, try_loss, try_mean_kl = f_surr_kl(*(all_input_values + [try_penalty]))
            logger.log('penalty %f => loss %f, mean kl %f' % (try_penalty, try_loss, try_mean_kl))
            if try_mean_kl < self.stepsize or (penalty_itr == max_penalty_itr - 1 and opt_params is None):
                opt_params = policy.get_param_values()
                penalty = try_penalty
                mean_kl = try_mean_kl

            if not self.adapt_penalty:
                break

            # decide scale factor on the first iteration
            if penalty_scale_factor is None or np.isnan(try_mean_kl):
                if try_mean_kl > self.stepsize or np.isnan(try_mean_kl):
                    # need to increase penalty
                    penalty_scale_factor = 2
                else:
                    # can shrink penalty
                    penalty_scale_factor = 0.5
            else:
                if penalty_scale_factor > 1 and try_mean_kl <= self.stepsize:
                    break
                elif penalty_scale_factor < 1 and try_mean_kl >= self.stepsize:
                    break
            try_penalty *= penalty_scale_factor


        policy.set_param_values(opt_params)
        loss_after = evaluate_cost(0)(opt_params)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)

        return merge_dict(opt_info, dict(
            cur_params=cur_params,
            opt_params=opt_params,
            penalty=penalty,
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
                'vf_params': samples_data['vf_params'],
                'all_obs': samples_data['all_obs'],
                'Q_est': samples_data['Q_est'],
                'penalty': opt_info['penalty'],
                #'paths': samples_data['paths'],
                'actions': samples_data['all_actions'],
                'pdists': samples_data['all_pdists'],
            }
            np.savez_compressed('%s/itr_%03d.npz' % (savedir, itr), **to_save)
