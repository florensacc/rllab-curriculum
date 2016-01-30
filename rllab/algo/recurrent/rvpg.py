import theano
import theano.tensor as TT
import numpy as np
from rllab.misc import logger, autoargs
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract, compile_function, new_tensor
from rllab.misc.tensor_utils import pad_tensor
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
            ndim=2+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.matrix('advantage')
        action_var = new_tensor(
            'action',
            ndim=3,
            dtype=mdp.action_dtype
        )
        log_prob = policy.get_log_prob_sym(obs_var, action_var)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - TT.mean(log_prob * advantage_var)
        grads = theano.grad(surr_obj, policy.get_params(trainable=True))
        input_list = [obs_var, advantage_var, action_var]

        f_log_prob = compile_function(
            inputs=input_list,
            outputs=log_prob
        )

        updates = self.update_method(grads, policy.get_params(trainable=True))

        f_update = compile_function(
            inputs=input_list,
            outputs=None,
            updates=updates,
        )
        return dict(
            f_update=f_update,
            f_log_prob=f_log_prob,
        )

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        logger.log("optimizing policy")
        f_update = opt_info["f_update"]
        paths = samples_data["paths"]

        max_path_length = max([len(path["advantages"]) for path in paths])

        # make all paths the same length (pad extra advantages with 0)
        obs = [path["observations"] for path in paths]
        obs = [pad_tensor(ob, max_path_length, ob[0]) for ob in obs]
        adv = [path["advantages"] for path in paths]
        adv = [pad_tensor(a, max_path_length, 0) for a in adv]
        actions = [path["actions"] for path in paths]
        actions = [pad_tensor(a, max_path_length, a[0]) for a in actions]

        # log_prob = opt_info["f_log_prob"](obs, adv, actions)
        # import ipdb; ipdb.set_trace()

        f_update(obs, adv, actions)
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
