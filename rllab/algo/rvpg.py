import theano
import theano.tensor as TT
import numpy as np
from rllab.misc import logger, autoargs
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract, compile_function, new_tensor
from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.first_order_method import FirstOrderMethod


class RVPG(BatchPolopt, FirstOrderMethod):
    """
    Recurrent Vanilla Policy Gradient.
    """

    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.inherit(FirstOrderMethod.__init__)
    def __init__(
            self,
            **kwargs):
        super(RVPG, self).__init__(**kwargs)
        FirstOrderMethod.__init__(self, **kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline):
        obs_var = new_tensor(
            'observations',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.vector('advantage')
        action_var = TT.matrix('action', dtype=mdp.action_dtype)
        reset_var = TT.ivector('episode_reset')
        log_prob = policy.get_log_prob_sym(obs_var, action_var, reset_var)
        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - TT.mean(log_prob * advantage_var)
        # grads = theano.grad(surr_obj, policy.params)
        # updates = self.update_method(surr_obj, policy.params)
        input_list = [obs_var, advantage_var, action_var, reset_var]
        # f_grads = compile_function(
        #     inputs=input_list,
        #     outputs=grads
        # )

        # grad_vars = [
        #     new_tensor('grad%d' % idx, ndim=grad.ndim, dtype=grad.dtype)
        #     for idx, grad in enumerate(grads)
        # ]
        updates = self.update_method(surr_obj, policy.params)

        f_update = compile_function(
            inputs=input_list,
            outputs=None,
            updates=updates,
        )
        # f_loss = compile_function(
        #     inputs=input_list,
        #     outputs=surr_obj,
        # )
        return dict(
            # f_grads=f_grads,
            f_update=f_update,
        )

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        logger.log("optimizing policy")
        # f_grads = opt_info["f_grads"]
        f_update = opt_info["f_update"]
        paths = samples_data["paths"]

        obs = samples_data["observations"]
        adv = samples_data["advantages"]
        actions = samples_data["actions"]
        # Construct binary array to indicate start of episodes
        episode_starts = np.zeros_like(adv)
        start_indices = [len(path["actions"]) for path in paths]
        episode_starts[[0] + np.cumsum(start_indices)[:-1]] = 1

        f_update(obs, adv, actions, episode_starts)

        # terminals = samples_data["terminals"]
        # actions = samples["actions"]
        # total_grads = None
        # total_path_length = 0
        # for path in paths:
        #     # print "lalala"
        #     # import sys
        #     # sys.exit(0)
        #     obs, adv, actions = extract(
        #         path,
        #         "observations", "advantages", "actions"
        #     )
        #     grads = f_grads(obs, adv, actions)
        #     path_length = len(adv)
        #     if total_grads is None:
        #         total_grads = [grad * path_length for grad in grads]
        #     else:
        #         total_grads = [total_grad + grad * path_length
        #                        for total_grad, grad in zip(total_grads, grads)]
        #     total_path_length += path_length
        # grads = [grad / total_path_length for grad in total_grads]
        # f_update(*grads)
        # inputs = extract(
        #     samples_data,
        #     "observations", "advantages", "actions"
        # )
        # loss_before = f_loss(*inputs)
        # f_update(*inputs)
        # loss_after = f_loss(*inputs)
        # logger.record_tabular("LossBefore", loss_before)
        # logger.record_tabular("LossAfter", loss_after)
        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data,
                         opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )
