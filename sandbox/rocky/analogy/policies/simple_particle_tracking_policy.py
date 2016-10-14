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
        particle_pos = particle_pos_n[self.env.target_id]
        action = np.clip(particle_pos - agent_pos, self.action_space.low, self.action_space.high)
        return action, dict()


if __name__ == "__main__":
    from sandbox.rocky.analogy.envs.simple_particle_env import SimpleParticleEnv
    from rllab.sampler.utils import rollout
    while True:
        env = SimpleParticleEnv(n_particles=6, seq_length=10)
        policy = SimpleParticleTrackingPolicy(env)
        rollout(env, policy, max_path_length=100, animated=True)
