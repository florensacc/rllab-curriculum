from sandbox.rocky.analogy.utils import unwrap
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.analogy.policies.base import AnalogyPolicy
import numpy as np
from sandbox.bradly.analogy.policy.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy


class MixturePolicy(AnalogyPolicy):
    def __init__(self, env, expert, novice):
        self.expert = expert
        self.novice = novice
        Policy.__init__(self, env_spec=env.spec)

    def get_action(self, obs):
        choice = np.random.randint(0, 1)
        if choice == 0:
            pol = self.expert
        else:
            pol = self.novice
        return pol.get_action(obs=obs)


if __name__ == "__main__":
    from sandbox.bradly.analogy.envs.conopt_particle_env import ConoptParticleEnv
    from rllab.sampler.utils import rollout

    while True:
        env = ConoptParticleEnv(particles_to_reach=3)
        expert = ConoptParticleTrackingPolicy(env)
        novice = ConoptParticleTrackingPolicy(env)
        policy = MixturePolicy(env, expert, novice)
        #policy = expert
        rollout(env, policy, max_path_length=120, animated=True)
