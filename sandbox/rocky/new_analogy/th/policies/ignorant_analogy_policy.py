from cached_property import cached_property
from torch import nn

from sandbox.rocky.th import tensor_utils
from sandbox.rocky.th.core.modules import MLP
from sandbox.rocky.th.distributions.utils import space_to_dist_dim, output_to_info, space_to_distribution
from sandbox.rocky.th.tensor_utils import Variable


class IgnorantAnalogyPolicy(nn.Module):
    def __init__(self, env_spec, hidden_sizes=(128, 128)):
        super().__init__()
        obs_dim = env_spec.observation_space.flat_dim
        action_flat_dim = space_to_dist_dim(env_spec.action_space)
        self.decoder = MLP(
            input_size=obs_dim,
            output_size=action_flat_dim,
            hidden_sizes=hidden_sizes,
        )
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space
        self.hidden_sizes = hidden_sizes
        self.demo_obs_pool = None

    def forward(self, demo_obs, obs):
        """
        :param demo_obs: observations along demo traj. Layout should be time * batch_size * dim
        :param obs: observations. Layout should be batch_size * dim
        :return:
        """
        obs_size = obs.size(2)
        nn_input = obs.view(-1, obs_size)
        nn_output = self.decoder(nn_input)
        return output_to_info(nn_output, self.action_space)

    @cached_property
    def distribution(self):
        return space_to_distribution(self.action_space)

    def reset(self, dones=None):
        pass

    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        flat_obs = Variable(tensor_utils.from_numpy(flat_obs).float(), volatile=True)
        nn_output = self.decoder(flat_obs)
        info_vars = output_to_info(nn_output, self.action_space)
        agent_infos = {k: tensor_utils.to_numpy(v) for k, v in info_vars.items()}
        actions = self.distribution.sample(agent_infos)
        return actions, agent_infos
