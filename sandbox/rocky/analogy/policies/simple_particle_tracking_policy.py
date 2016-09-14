from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.analogy.policies.base import AnalogyPolicy
import numpy as np


class SimpleParticleTrackingPolicy(AnalogyPolicy):
    def __init__(self, env):
        if isinstance(env, TfEnv):
            env = env.wrapped_env
        self.env = env
        Policy.__init__(self, env_spec=env.spec)

    def get_action(self, obs):
        agent_pos, particle_pos_n = obs
        particle_pos = particle_pos_n[self.env.target_id]
        action = np.clip(particle_pos - agent_pos, self.action_space.low, self.action_space.high)
        return action, dict()


if __name__ == "__main__":
    from sandbox.rocky.analogy.envs.simple_particle_env import SimpleParticleEnv
    from rllab.sampler.utils import rollout
    while True:
        env = SimpleParticleEnv()
        policy = SimpleParticleTrackingPolicy(env)
        rollout(env, policy, max_path_length=10, animated=True)
