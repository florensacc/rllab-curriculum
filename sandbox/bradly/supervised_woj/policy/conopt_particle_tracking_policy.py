from sandbox.rocky.analogy.utils import unwrap
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.analogy.policies.base import AnalogyPolicy
import numpy as np
from cached_property import cached_property


class ConoptParticleTrackingPolicy(AnalogyPolicy):
    def __init__(self, env):
        self.env = unwrap(env)
        Policy.__init__(self, env_spec=env.spec)

    @cached_property
    def agent_coords_offset(self):
        return self.env.model.site_names.index(b'object') * 3

    def target_coords_offset(self):
        return self.env.model.site_names.index(b'target') * 3

    def get_action(self, obs):
        # print(obs[2].shape)
        agent_pos = obs[2][self.agent_coords_offset:self.agent_coords_offset + 2]  # does this need a leading 0 index?
        target_pos = obs[2][self.target_coords_offset():self.target_coords_offset() + 2]
        action = target_pos - agent_pos
        action -= np.squeeze(obs[1]) * 0.05
        return self.env.action_space.unflatten(action), dict()


if __name__ == "__main__":
    from sandbox.bradly.supervised_woj.envs.conopt_particle_env import ConoptParticleEnv
    from rllab.sampler.utils import rollout

    while True:
        env = ConoptParticleEnv(particles_to_reach=3)
        policy = ConoptParticleTrackingPolicy(env)
        rollout(env, policy, max_path_length=120, animated=True)
