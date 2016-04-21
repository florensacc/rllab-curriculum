from __future__ import print_function
from __future__ import absolute_import

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import tensor_utils
import numpy as np


class PriorHallucinator(Serializable):
    """
    Hallucinate additional samples for the latent variables by naive ancestral sampling.
    """

    def __init__(self, policy, n_samples=5, self_normalize=False):
        """
        :param policy:
        :param n_samples:
        :param self_normalize: whether to normalize the importance weights so that the weights for the same true
        sample sum to one
        :return:
        """
        Serializable.quick_init(self, locals())
        self.policy = policy
        self.n_samples = n_samples
        self.self_normalize = self_normalize

    def hallucinate(self, samples_data):
        # we'd like to extend the experience with extra trajectories
        observations, actions, advantages, env_infos, agent_infos = \
            ext.extract(
                samples_data,
                "observations", "actions", "advantages", "env_infos", "agent_infos"
            )
        all_importance_weights = [
            np.ones_like(advantages)
        ]
        all_observations = [
            observations
        ]
        all_actions = [
            actions
        ]
        all_advantages = [
            advantages
        ]
        all_env_infos = [
            env_infos
        ]
        all_agent_infos = [
            agent_infos
        ]
        dist = self.policy.distribution
        old_logli = dist.log_likelihood(actions, agent_infos)
        for _ in xrange(self.n_samples):
            new_actions, new_agent_infos = self.policy.get_actions(observations)
            # We'd need to compute the importance ratio. This is given by p(a|h_new) / p(a|h_old)
            all_importance_weights.append(
                np.exp(dist.log_likelihood(actions, new_agent_infos) - old_logli)
            )
            all_observations.append(observations)
            all_actions.append(actions)
            all_advantages.append(advantages)
            all_env_infos.append(env_infos)
            all_agent_infos.append(agent_infos)
        if self.self_normalize:
            all_importance_weights = np.asarray(all_importance_weights)
            all_importance_weights = all_importance_weights / (np.sum(all_importance_weights, axis=0, keepdims=True)
                                                               + 1e-8)
        return dict(
            observations=tensor_utils.concat_tensor_list(all_observations),
            actions=tensor_utils.concat_tensor_list(all_actions),
            advantages=tensor_utils.concat_tensor_list(all_advantages),
            env_infos=tensor_utils.concat_tensor_dict_list(all_env_infos),
            agent_infos=tensor_utils.concat_tensor_dict_list(all_agent_infos),
            importance_weights=tensor_utils.concat_tensor_list(all_importance_weights),
        )
