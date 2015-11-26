import tensorfuse.tensor as TT
from rllab.misc import logger, autoargs
from rllab.misc.overrides import overrides
from rllab.misc.ext import compile_function
from rllab.algo.util import center_advantages
from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.first_order_method import FirstOrderMethod
import numpy as np


class RSVPG(BatchPolopt, FirstOrderMethod):
    """
    Risk-Seeking Vanilla Policy Gradient.
    """

    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.inherit(FirstOrderMethod.__init__)
    @autoargs.arg("q_threshold", type=float,
                  help="Theshold to only take the best q portion of episodes")
    def __init__(
            self,
            q_threshold=0.2,
            **kwargs):
        self.q_threshold = q_threshold
        super(RSVPG, self).__init__(**kwargs)
        FirstOrderMethod.__init__(self, **kwargs)

    @overrides
    def init_opt(self, mdp, policy, vf):
        input_var = TT.tensor(
            'input',
            ndim=len(mdp.observation_shape)+1,
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.vector('advantage')
        action_var = TT.matrix('action', dtype=mdp.action_dtype)
        log_prob = policy.get_log_prob_sym(input_var, action_var)
        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - TT.mean(log_prob * advantage_var)
        updates = self.update_method(surr_obj, policy.params)
        input_list = [input_var, advantage_var, action_var]
        f_update = compile_function(
            inputs=input_list,
            outputs=None,
            updates=updates,
        )
        f_loss = compile_function(
            inputs=input_list,
            outputs=surr_obj,
        )
        return dict(
            f_update=f_update,
            f_loss=f_loss,
        )

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        """
        Matlab code
        sample N trajectories x^1_1...x^N_t
        for i = 1:N
           r_i = sum_t(r(x^i_t))
        end
        [sr ] = sort(r)
        ind = indices of best 20% of r
        q = r_ind(0)
        grad = 1/(length(ind)) * sum_{i in ind} (
            grad_log(x^i_1,...,x^i_t) * (r_i - q) )

        This grad formula is equivalent to the following:
        grad = 1/(length(ind)) * sum_{i in ind} sum_{t} (
            grad_log(a^i_t | x^i_t) * (r_i - q) )
        """

        returns = np.array([
            path["returns"][0] for path in samples_data["paths"]
        ])
        N = len(returns)
        # sort from largest to smallest
        sorted_ids = np.argsort(-returns)
        # get the best (p*100)% of returns
        best_ids = sorted_ids[:int(N*self.q_threshold)]
        baseline = returns[best_ids[-1]]

        # collect the best (p*100)% of training data
        observations = []
        advantages = []
        actions = []

        for idx in best_ids:
            path = samples_data["paths"][idx]
            ret = path["returns"][0]
            observations.append(path["observations"])
            actions.append(path["actions"])
            # the whole path use the same advantage, according to the formula
            # above
            advantages.append(np.repeat(ret - baseline, len(path["actions"])))

        observations = np.vstack(observations)
        if self.center_adv:
            advantages = center_advantages(np.concatenate(advantages))
        actions = np.vstack(actions)

        logger.log("optimizing policy")
        f_update = opt_info["f_update"]
        f_loss = opt_info["f_loss"]
        inputs = [observations, advantages, actions]
        loss_before = f_loss(*inputs)
        f_update(*inputs)
        loss_after = f_loss(*inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)
        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, vf, samples_data, opt_info):
        return dict(
            itr=itr,
            policy=policy,
            vf=vf,
            mdp=mdp,
            observations=samples_data["observations"],
            advantages=samples_data["advantages"],
            actions=samples_data["actions"],
            paths=samples_data["paths"],
        )
