# from rllab.algos.npo import NPO
from sandbox.adam.gpar.algos.npo import NPO
# from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.adam.gpar.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides


class TRPO(NPO):
    """
    Trust Region Policy Optimization.
    CHANGES:
    1. Different optimizer file -- speedups.
    2. Refactored optimize policy -- move logging inside optimizer to remove
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

    def prepare_opt_inputs(self, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        print("obs dtype: ", samples_data["observations"].dtype)
        print("act dtype: ", samples_data["actions"].dtype)
        print("adv dtype: ", samples_data["advantages"].dtype)
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        return all_input_values

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = self.prepare_opt_inputs(samples_data)
        self.optimizer.optimize(all_input_values)
        return dict()


