from sandbox.bradly.third_person.discriminators.discriminator import DomainConfusionDiscriminator
from sandbox.bradly.third_person.envs.conopt_particle_env import ConoptParticleEnv
from sandbox.bradly.third_person.policy.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy

from rllab.sampler.utils import rollout
from rllab.misc import tensor_utils
import time


class CyberPunkTrainer:
    def __init__(self, env, policy, im_width, im_height, im_channels=3):
        dim_input = [im_width, im_height, im_channels]
        self.disc = DomainConfusionDiscriminator(input_dim=dim_input, output_dim_class=2, output_dim_dom=2)

        self.policy = policy
        self.env = env
        self.expert = ConoptParticleTrackingPolicy(self.env)
        self.horizon = 100

    def train(self, itr, n_samps=32):
        expert_samps = self.get_some_expert_samples(n_samps)
        expert_failures = 0
        on_policy_samps = 0
        data_batch, targets_batch = self.create_batch()
        self.disc.train(data_batch, targets_batch)
        #self.policy.optimize_policy(self, itr, samples_data):

    def create_batch(self):
        pass

    def get_some_on_policy_samps(self, n_samps):
        paths = []
        for sample_iter in range(0, n_samps):
            paths.append(self.on_policy_rollout(self.policy, self.disc, self.env, self.horizon))

    def get_some_expert_samples(self, n_samps):
        paths = []
        for samp_iter in range(0, n_samps):
            paths.append(rollout(self.env, self.expert, max_path_length=self.horizon))
            return paths

    def get_some_bad_samps(self, n_samps):
        pass

    def on_policy_rollout(self, agent, encoder, env, max_path_length, animated=False, speedup=1):
        observations = []
        encoded_observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        o = env.reset()
        path_length = 0
        if animated:
            env.render()
        while path_length < max_path_length:
            a, agent_info = agent.get_action(o)
            encoded_o = encoder.encode(o)
            encoded_observations.append(encoded_o)
            next_o, r, d, env_info = env.step(a)
            r = encoder.get_reward(o)
            observations.append(env.observation_space.flatten(o))
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
            true_observations=tensor_utils.stack_tensor_list(observations),
            observations = tensor_utils.stack_tensor_list(encoded_observations),
            actions=tensor_utils.stack_tensor_list(actions),
            rewards=tensor_utils.stack_tensor_list(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        )

