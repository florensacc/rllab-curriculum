# from rllab.algos.npo import NPO
from sandbox.adam.gpar.algos.npo import NPO
# from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.adam.gpar.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides


class TRPO(NPO):
    """
    Trust Region Policy Optimization
    CHANGES:
    1. Class inheritance.
    2. Different optimizer file -- speedups.
    3. Refactored optimize policy -- move logging inside optimizer to remove
       redundant calculations.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPO, self).__init__(optimizer=optimizer, **kwargs)


    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        # loss_before = self.optimizer.loss(all_input_values)
        # mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)
        # mean_kl = self.optimizer.constraint_val(all_input_values)
        # loss_after = self.optimizer.loss(all_input_values)
        # logger.record_tabular('LossBefore', loss_before)
        # logger.record_tabular('LossAfter', loss_after)
        # logger.record_tabular('MeanKLBefore', mean_kl_before)
        # logger.record_tabular('MeanKL', mean_kl)
        # logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()
