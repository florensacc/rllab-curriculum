import tensorfuse as theano
import tensorfuse.tensor as TT
import lasagne.updates
from rllab.sampler import parallel_sampler
from rllab.misc import logger, autoargs
from rllab.misc.ext import extract, compact, merge_dict
from rllab.misc.console import mkdir_p
from rllab.algo.base import RLAlgorithm
from functools import partial
import numpy as np
import cPickle as pickle


def new_opt_vals(policy, input_var, adv_var, action_var, ref_input_var, ref_policy):
    assert policy.input_var is input_var
    log_prob = policy.get_log_prob_sym(action_var)
    # formulate as a minimization problem
    # The gradient of the surrogate objective is the policy gradient
    surrogate_obj = - TT.mean(log_prob * adv_var)
    # We would need to calculate the empirical fisher information matrix
    # This can be done as follows (though probably not the most efficient way):
    # Since I(theta) = d^2 KL(p(theta)||p(theta')) / d theta^2
    #                  (evaluated at theta' = theta),
    # we can get I(theta) by calculating the hessian of KL(p(theta)||p(theta'))
    mean_kl = TT.mean(policy.kl(policy.pdist_var, ref_policy.pdist_var))
    # Here, we need to ensure that all the parameters are flattened
    emp_fishers = theano.gradient.hessian(mean_kl, wrt=policy.params)
    grads = theano.grad(surrogate_obj, wrt=policy.params)
    # Is there a better name...
    fisher_grads = []
    for emp_fisher, grad in zip(emp_fishers, grads):
        fisher_grads.append(TT.nlinalg.matrix_inverse(emp_fisher + TT.eye(emp_fisher.shape[0])).dot(grad))
    return surrogate_obj, fisher_grads


def get_train_vars(policy):
    input_var = policy.input_var
    adv_var = TT.vector('advantage')
    action_var = policy.new_action_var('action')
    ref_policy = pickle.loads(pickle.dumps(policy))
    ref_input_var = ref_policy.input_var
    return dict(
        input_var=input_var,
        adv_var=adv_var,
        action_var=action_var,
        ref_input_var=ref_input_var,
        ref_policy=ref_policy,
    )


def to_input_var_list(input_var, adv_var, action_var, ref_input_var, ref_policy):
    return [input_var, adv_var, action_var, ref_input_var]


def get_all_inputs(samples_data):
    return extract(
        samples_data,
        "all_obs", "all_advantages", "all_actions", "all_obs"
    )


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **compact(kwargs))
    else:
        raise NotImplementedError


class NPG(RLAlgorithm):
    """
    Natural Policy Gradient.
    Need access to the empirical fisher information matrix
    """

    @autoargs.arg('max_path_length', type=int, help='Maximum length of a rollout.')
    @autoargs.arg('batch_size', type=int, help='Size for each batch.')
    @autoargs.arg('update_method', type=str, help='Update method.')
    @autoargs.arg('n_itr', type=int, help='Number of iterations.')
    @autoargs.arg('exp_name', type=str, help='Name of the experiment.')
    @autoargs.arg('discount', type=float, help='Discount.')
    @autoargs.arg('gae_lambda', type=float, help='Lambda used for generalized advantage estimation.')
    @autoargs.arg('learning_rate', type=float, help='Learning rate.')
    @autoargs.arg('plot', type=bool, help='Plot a test rollout per iteration.')
    @autoargs.arg('whiten_advantage', type=bool, help='Whether to rescale and center the advantage values.')
    @autoargs.arg('save_snapshot', type=bool, help='Whether to save parameters for each iteration.')
    def __init__(
            self,
            max_path_length=np.inf,
            batch_size=5000,
            update_method='sgd',
            learning_rate=None,
            n_itr=500,
            exp_name='npg',
            discount=0.99,
            gae_lambda=1,
            plot=False,
            whiten_advantage=True,
            save_shapshot=True
            ):
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.update_method = parse_update_method(
            update_method,
            learning_rate=learning_rate
        )
        self.n_itr = n_itr
        self.exp_name = exp_name
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.whiten_advantage = whiten_advantage
        self.save_snapshot = save_snapshot

    def start_worker(self, mdp, policy, vf):
        parallel_sampler.populate_task(mdp, policy)

    def train(self, mdp, policy, vf, **kwargs):
        savedir = 'data/%s' % (self.exp_name)
        logger.add_file_output(savedir + '/log.txt')
        opt_info = self.init_opt(mdp, policy, vf)
        self.start_worker(mdp, policy, vf)
        logger.push_prefix('[%s] | ' % (self.exp_name))

        for itr in xrange(self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            samples_data = self.obtain_samples(itr, mdp, policy, vf)
            opt_info = self.optimize_policy(itr, policy, samples_data, opt_info)
            if self.save_snapshot:
                self.perform_save_snapshot(itr, samples_data, opt_info)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

        logger.remove_file_output(savedir + '/log.txt')
        logger.pop_prefix()

    def obtain_samples(self, itr, mdp, policy, vf):
        results = parallel_sampler.request_samples_stats(
            itr=itr,
            mdp=mdp,
            policy=policy,
            vf=vf,
            samples_per_itr=self.batch_size,
            max_path_length=self.max_path_length,
            discount=self.discount,
            gae_lambda=self.gae_lambda,
            record_states=False,
        )
        vf.fit(results["paths"])
        if self.whiten_advantage:
            def center_qval(qval):
                return (qval - np.mean(qval)) / (qval.std() + 1e-8)

            results["all_advantages"] = center_qval(results["all_advantages"])
        results["vf_params"] = vf.get_param_values()
        return results

    def init_opt(self, mdp, policy, vf):
        train_vars = get_train_vars(policy)
        surr_obj, fisher_grads = new_opt_vals(policy, **train_vars)
        updates = self.update_method(fisher_grads, policy.params)
        input_list = to_input_var_list(**train_vars)
        f_update = theano.function(
            inputs=input_list,
            outputs=surr_obj,
            updates=updates,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        f_loss = theano.function(
            inputs=input_list,
            outputs=surr_obj,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        return dict(
            f_update=f_update,
            f_loss=f_loss,
            ref_policy=train_vars["ref_policy"],
        )

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        cur_params = policy.get_param_values()
        logger.log("optimizing policy")
        f_update = opt_info["f_update"]
        f_loss = opt_info["f_loss"]
        all_inputs = get_all_inputs(samples_data)
        logger.log("computing loss before")
        loss_before = f_loss(*all_inputs)
        # Need to ensure this
        opt_info["ref_policy"].set_param_values(cur_params)
        f_update(*all_inputs)
        loss_after = f_loss(*all_inputs)
        logger.record_tabular("Loss Before", loss_before)
        logger.record_tabular("Loss After", loss_after)
        opt_params = policy.get_param_values()
        return merge_dict(opt_info, dict(
            cur_params=cur_params,
            opt_params=opt_params,
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
                'all_advantages': samples_data['all_advantages'],
                'actions': samples_data['all_actions'],
            }
            np.savez_compressed('%s/itr_%03d.npz' % (savedir, itr), **to_save)
