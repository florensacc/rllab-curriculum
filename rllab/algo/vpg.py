import tensorfuse as theano
import tensorfuse.tensor as TT
import lasagne.updates
from rllab.sampler import parallel_sampler
from rllab.misc import logger, autoargs
from rllab.misc.ext import extract, compact
from rllab.algo.base import RLAlgorithm
from functools import partial
import numpy as np

# Vanilla Policy Gradient


def new_surrogate_obj(policy, input_var, adv_var, action_var):
    assert policy.input_var is input_var
    log_prob = policy.get_log_prob_sym(action_var)
    # formulate as a minimization problem
    # The gradient of the surrogate objective is the policy gradient
    surrogate_obj = - TT.mean(log_prob * adv_var)
    return surrogate_obj


def get_train_vars(policy):
    input_var = policy.input_var
    adv_var = TT.vector('advantage')
    action_var = policy.new_action_var('action')
    return dict(
        input_var=input_var,
        adv_var=adv_var,
        action_var=action_var,
    )


def to_input_var_list(input_var, adv_var, action_var):
    return [input_var, adv_var, action_var]


def get_all_inputs(samples_data):
    return extract(
        samples_data,
        "all_obs", "all_advantages", "all_actions"
    )


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **compact(kwargs))
    else:
        raise NotImplementedError


class VPG(RLAlgorithm):
    """
    Vanilla Policy Gradient.
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
    def __init__(
            self,
            max_path_length=np.inf,
            batch_size=5000,
            update_method='adam',
            learning_rate=None,
            n_itr=500,
            exp_name='vpg',
            discount=0.99,
            gae_lambda=1,
            plot=False
            ):
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.update_method = parse_update_method(
            update_method,
            learning_rate=learning_rate
        )
        self.learning_rate = learning_rate
        self.n_itr = n_itr
        self.exp_name = exp_name
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot

    def start_worker(self, mdp, policy, vf):
        parallel_sampler.populate_task(mdp, policy)

    def train(self, mdp, policy, vf, **kwargs):
        opt_info = self.init_opt(mdp, policy, vf)
        self.start_worker(mdp, policy, vf)
        logger.push_prefix('[%s] | ' % (self.exp_name))

        for itr in xrange(self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            samples_data = self.obtain_samples(itr, mdp, policy, vf)
            opt_info = self.optimize_policy(itr, policy, samples_data, opt_info)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

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
        return results

    def init_opt(self, mdp, policy, vf):
        train_vars = get_train_vars(policy)
        surr_obj = new_surrogate_obj(policy, **train_vars)
        updates = self.update_method(surr_obj, policy.params)
        input_list = to_input_var_list(**train_vars)
        f_update = theano.function(
            inputs=input_list,
            outputs=theano.grad(surr_obj, policy.params),
            updates=updates,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        #f_grad = theano.function(
        #    inputs=input_list,
        #    outputs=theano.grad(surr_obj),#, policy.params[0]),
        #    on_unused_input='ignore',
        #    allow_input_downcast=True
        #)
        f_loss = theano.function(
            inputs=input_list,
            outputs=surr_obj,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        #import ipdb; ipdb.set_trace()
        return dict(
            f_update=f_update,
            #f_grad=f_grad,
            f_loss=f_loss
        )

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        logger.log("optimizing policy")
        f_update = opt_info["f_update"]
        f_loss = opt_info["f_loss"]
        all_inputs = get_all_inputs(samples_data)
        f_update(*all_inputs)
        return opt_info
