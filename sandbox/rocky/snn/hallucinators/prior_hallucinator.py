from __future__ import print_function
from __future__ import absolute_import

from rllab.core.serializable import Serializable
import numpy as np


class PriorHallucinator(Serializable):
    """
    Hallucinate additional samples for the latent variables by naive ancestral sampling.
    """

    def __init__(self, env_spec, policy, n_hallucinate_samples=5):
        """
        :param policy:
        :param n_hallucinate_samples:
        :return:
        """
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.policy = policy
        self.n_hallucinate_samples = n_hallucinate_samples

    def hallucinate(self, samples_data):
        # we'd like to extend the experience with extra trajectories
        observations = samples_data["observations"]
        actions = samples_data["actions"]
        agent_infos = samples_data["agent_infos"]
        h_samples = []
        old_logli = self.policy.log_likelihood(actions, agent_infos, action_only=True)
        for _ in xrange(self.n_hallucinate_samples):
            new_actions, new_agent_infos = self.policy.get_actions(observations)
            new_logli = self.policy.log_likelihood(actions, new_agent_infos, action_only=True)
            # We'd need to compute the importance ratio. This is given by p(a|h_new) / p(a|h_old)
            h_samples.append(
                dict(
                    samples_data,
                    importance_weights=np.exp(new_logli - old_logli),
                    agent_infos=new_agent_infos,
                )
            )
        return h_samples
