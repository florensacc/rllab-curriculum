from sandbox.rocky.analogy.utils import unwrap
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.analogy.policies.base import AnalogyPolicy
import numpy as np


class SimpleParticleTrackingPolicy(AnalogyPolicy):
    def __init__(self, env):
        self.env = unwrap(env)
        Policy.__init__(self, env_spec=env.spec)

    def get_action(self, obs):
        agent_pos = self.env.agent_pos
        particle_pos_n = self.env.particles
        particle_pos = particle_pos_n[self.env.target_ids[self.env.target_id_idx]]
        action = np.clip(particle_pos - agent_pos, self.action_space.low, self.action_space.high)
        #if np.linalg.norm(particle_pos - agent_pos) < 0.01:
        #    self.env.target_id = 2
        return action, dict()


if __name__ == "__main__":
    from sandbox.bradly.analogy.envs.simple_particle_env import SimpleParticleEnv
    from rllab.sampler.utils import rollout
    while True:
        env = SimpleParticleEnv(n_particles=2)
        policy = SimpleParticleTrackingPolicy(env)
        rollout(env, policy, max_path_length=20, animated=True)
        #env.target_id = 2
        #rollout(env, policy, max_path_length=10, animated=True)
