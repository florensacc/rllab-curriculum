# get on policy samples success + failure
# get expert samples success + failure
# train domain invariant disc and see if it works

from sandbox.bradly.third_person.discriminators.discriminator import DomainConfusionDiscriminator
from rllab.sampler.utils import rollout
import numpy as np
from sandbox.bradly.third_person.policy.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy
from sandbox.bradly.third_person.envs.conopt_particle_env import ConoptParticleEnv


class IdeaClass:
    def __init__(self, policy_env, expert_env, expert_policy, random_policy):
        self.policy_env = policy_env
        self.expert_env = expert_env
        self.expert_policy = expert_policy
        self.random_policy = random_policy
        self.batch_size = 32
        self.horizon = 100
        self.im_height = 50
        self.im_width = 50
        self.im_channels = 3
        #self.disc = DomainConfusionDiscriminator(input_dim=[50, 50, 3], output_dim_class=2, output_dim_dom=2)

    def get_rollouts(self, num_trajs, policy, env, dom, cls):
        paths = []
        for iter_step in range(0, num_trajs):
            paths.append(non_stupid_rollout(env, policy, self.horizon, animated=True))
            paths[-1]['dom'] = dom
            paths[-1]['class'] = cls
        asdsf
        return paths

    def train(self):
        pass

    def gen_data(self):
        num_trajs = 1000
        e_10 = np.zeros((2,))
        e_10[0] = 1
        e_01 = np.zeros((2,))
        e_01[1] = 1
        expert_dom_0 = self.get_rollouts(1000, self.expert_policy, self.expert_env, e_10, cls=e_10)
        expert_dom_1 = self.get_rollouts(1000, self.expert_policy, self.policy_env, dom=e_01, cls=e_10)
        novice_dom_0 = self.get_rollouts(1000, self.random_policy, self.expert_env, dom=e_10, cls=e_01)
        novice_dom_1 = self.get_rollouts(1000, self.random_policy, self.policy_env, dom=e_01, cls=e_01)
        data_matrix = np.zeros(shape=(num_trajs*self.horizon*4, self.im_height, self.im_width, self.im_channels))
        class_matrix = np.zeros(shape=(num_trajs*self.horizon*4, 2))
        dom_matrix = np.zeros(shape=(num_trajs*self.horizon*4, 2))
        mega_paths = expert_dom_0 + expert_dom_1 + novice_dom_0 + novice_dom_1
        iter_step = 0
        for path in mega_paths:
            data_matrix[iter_step, :, :, :] = path['observations'][3]
            class_matrix[iter_step, :, :, :] = path['class']
            dom_matrix[iter_step, :, :, :] = path['dom']
        np.save('data', data_matrix)
        np.save('classes', class_matrix)
        np.save('domains', dom_matrix)

    def load_data(self):
        pass

    def main(self):

        n_epochs = 5
        for iter_step in range(0, n_epochs):
            self.disc.train()




from rllab.misc import tensor_utils
import time


def non_stupid_rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render(close=True)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )



expert_env = ConoptParticleEnv()
expert_pol = ConoptParticleTrackingPolicy(expert_env)

cls = IdeaClass(expert_env, expert_env, expert_pol, expert_pol)
cls.get_rollouts(1, expert_pol, expert_env, 0, 0)