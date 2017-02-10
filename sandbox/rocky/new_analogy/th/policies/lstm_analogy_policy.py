from cached_property import cached_property
from torch import nn
import torch

from sandbox.rocky.th import tensor_utils
from sandbox.rocky.th.core.modules import MLP
from sandbox.rocky.th.distributions.utils import space_to_dist_dim, output_to_info, space_to_distribution
from sandbox.rocky.th.tensor_utils import Variable
import numpy as np


class LSTMAnalogyPolicy(nn.Module):
    def __init__(self, env_spec, rnn_hidden_size=32, hidden_sizes=(128, 128), rnn_num_layers=1):
        super().__init__()
        obs_dim = env_spec.observation_space.flat_dim
        action_flat_dim = space_to_dist_dim(env_spec.action_space)
        self.encoder = nn.LSTM(
            input_size=obs_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=False,
        )
        self.decoder = MLP(
            input_size=(obs_dim + rnn_hidden_size),
            output_size=action_flat_dim,
            hidden_sizes=hidden_sizes,
        )
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.hidden_sizes = hidden_sizes
        self.demo_obs_pool = None
        self.demo_embedding = None

    def init_hidden(self, batch_size, volatile=False):
        weight = next(self.parameters()).data
        h0 = Variable(
            weight.new(self.rnn_num_layers, batch_size, self.rnn_hidden_size).zero_(),
            volatile=volatile
        )
        c0 = Variable(
            weight.new(self.rnn_num_layers, batch_size, self.rnn_hidden_size).zero_(),
            volatile=volatile
        )
        return (h0, c0)

    def forward(self, demo_obs, obs):
        """
        :param demo_obs: observations along demo traj. Layout should be time * batch_size * dim
        :param obs: observations. Layout should be batch_size * dim
        :return:
        """
        # only take the last time step, as the encoding of the entire trajectory
        demo_embedding = self.get_embedding(demo_obs)
        demo_batch_size = demo_embedding.size(0)
        per_demo_batch_size = obs.size(1)
        embedding_size = demo_embedding.size(1)
        obs_size = obs.size(2)

        demo_embedding = demo_embedding \
            .unsqueeze(1) \
            .expand(demo_batch_size, per_demo_batch_size, embedding_size)

        nn_input = torch.cat([obs, demo_embedding], 2).view(-1, obs_size + embedding_size)

        nn_output = self.decoder(nn_input)

        return output_to_info(nn_output, self.action_space)

    def get_embedding(self, demo_obs, volatile=False):
        batch_size = demo_obs.size(1)
        outputs, _ = self.encoder(demo_obs, self.init_hidden(batch_size, volatile=volatile))
        demo_embedding = outputs[-1]
        return demo_embedding

    @cached_property
    def distribution(self):
        return space_to_distribution(self.action_space)

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.cast['bool'](dones)
        cnt = int(np.sum(dones))
        if cnt > 0:
            demo_ids = np.random.choice(len(self.demo_obs_pool), size=cnt, replace=True)
            demo_obs = np.transpose(self.demo_obs_pool[demo_ids], (1, 0, 2))
            demo_obs = Variable(tensor_utils.from_numpy(demo_obs).float(), volatile=True)
            demo_embedding = self.get_embedding(demo_obs, volatile=True)
            if self.demo_embedding is None or self.demo_embedding.size(0) != len(dones):
                self.demo_embedding = demo_embedding
            else:
                mask = Variable(
                    tensor_utils.from_numpy(np.cast['uint8'](dones)).byte()
                ).unsqueeze(1).expand_as(self.demo_embedding)
                self.demo_embedding[mask] = demo_embedding

    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        flat_obs = Variable(tensor_utils.from_numpy(flat_obs).float(), volatile=True)
        nn_input = torch.cat([flat_obs, self.demo_embedding], 1)
        nn_output = self.decoder(nn_input)
        info_vars = output_to_info(nn_output, self.action_space)
        agent_infos = {k: tensor_utils.to_numpy(v) for k, v in info_vars.items()}
        actions = self.distribution.sample(agent_infos)
        return actions, agent_infos
