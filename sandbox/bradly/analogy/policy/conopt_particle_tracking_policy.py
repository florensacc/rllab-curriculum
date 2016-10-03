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
        target_id = self.env.conopt_scenario.task_id
        return self.env.model.site_names.index(('target%d' % target_id).encode()) * 3

    def get_action(self, obs):
        agent_pos = obs[2][0, self.agent_coords_offset:self.agent_coords_offset + 2]
        target_pos = obs[2][0, self.target_coords_offset():self.target_coords_offset() + 2]
        action = target_pos - agent_pos
        action -= np.squeeze(obs[1]) * 0.05
        #if np.linalg.norm(agent_pos - target_pos) < 0.05:
        #    #print('kay')
        #    potential_targ = self.env.conopt_scenario.task_id + 1
        #    new_targ = min(potential_targ, 1)
        #    self.env.conopt_scenario.task_id = new_targ
        #    #self.target_coords_offset()
        #return self.env.action_space.unflatten(action), dict()


if __name__ == "__main__":
    from sandbox.bradly.analogy.envs.conopt_particle_env import ConoptParticleEnv
    from rllab.sampler.utils import rollout

    while True:
        env = ConoptParticleEnv()
        policy = ConoptParticleTrackingPolicy(env)
        rollout(env, policy, max_path_length=100, animated=True)
