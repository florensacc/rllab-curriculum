from rllab.core.serializable import Serializable
import numpy as np
from sandbox.carlos_snn.hallucinators.base import BaseHallucinator


class PriorHallucinator(BaseHallucinator, Serializable):
    """
    Hallucinate additional samples for the latents variables by naive ancestral sampling.
    """

    def hallucinate(self, samples_data):  # here samples data have already been processed, so it has all Adv, Ret,..
        # we'd like to extend the experience with extra trajectories
        observations = samples_data["observations"]
        actions = samples_data["actions"]
        agent_infos = samples_data["agent_infos"]
        h_samples = []
        old_logli = self.policy.log_likelihood(actions, agent_infos, action_only=True)
        for _ in range(self.n_hallucinate_samples):
            self.policy.reset()  # this will sample a new fixed latents if needed (resample False)
            new_actions, new_agent_infos = self.policy.get_actions(observations)
            new_logli = self.policy.log_likelihood(actions, new_agent_infos, action_only=True)
            # We'd need to compute the importance ratio. This is given by p(a|h_new) / p(a|h_old)
            h_samples.append(
                dict(
                    samples_data,
                    importance_weights=np.exp(new_logli - old_logli),
                    agent_infos=new_agent_infos,   # overwrites this original agent_infos in samples_data
                )
            )
        return h_samples
