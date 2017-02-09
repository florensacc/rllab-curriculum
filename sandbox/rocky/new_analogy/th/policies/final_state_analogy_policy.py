from cached_property import cached_property
from torch import nn

from rllab.core.serializable import Serializable
from sandbox.rocky.th import tensor_utils
from sandbox.rocky.th.core.module_powered import ModulePowered
from sandbox.rocky.th.core.modules import MLP
from sandbox.rocky.th.distributions.utils import space_to_dist_dim, output_to_info, space_to_distribution
import torch.nn.functional as F
import numpy as np
import torch


class FinalStateAnalogyPolicy(ModulePowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(128, 128),
            embedding_hidden_sizes=(),
            nonlinearity=F.relu,
            batch_norm=False,
    ):
        Serializable.quick_init(self, locals())
        super().__init__()
        obs_dim = env_spec.observation_space.flat_dim
        action_flat_dim = space_to_dist_dim(env_spec.action_space)
        if len(embedding_hidden_sizes) == 0:
            embedding_size = obs_dim
        else:
            embedding_size = embedding_hidden_sizes[-1]
        if len(embedding_hidden_sizes) > 0:
            self.encoder = MLP(
                input_size=obs_dim,
                output_size=embedding_hidden_sizes[-1],
                hidden_sizes=embedding_hidden_sizes[:-1],
                nonlinearity=nonlinearity,
                output_nonlinearity=nonlinearity,
                batch_norm=batch_norm,
                batch_norm_final=batch_norm,
            )
        else:
            self.encoder = lambda x: x
        self.decoder = MLP(
            input_size=obs_dim + embedding_size,
            output_size=action_flat_dim,
            hidden_sizes=hidden_sizes,
            nonlinearity=nonlinearity,
            batch_norm=batch_norm,
            batch_norm_final=False,
        )
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space
        self.hidden_sizes = hidden_sizes
        self.embedding_hidden_sizes = embedding_hidden_sizes
        self.embedding_size = embedding_size
        self.demo_obs_pool = None
        self.demo_embedding = None

    def forward(self, demo_paths, obs):
        """
        :param demo_obs: observations along demo traj. Layout should be time * batch_size * dim
        :param obs: observations. Layout should be batch_size * dim
        :return:
        """
        demo_batch_size = len(demo_paths)
        per_demo_batch_size = obs.size(1)
        # only take the final states
        demo_obs = np.asarray([p["observations"][-1] for p in demo_paths])
        demo_obs = tensor_utils.variable(demo_obs, dtype='float', requires_grad=False)
        demo_embedding = self.encoder(demo_obs)
        demo_embedding = demo_embedding.unsqueeze(1) \
            .expand(demo_batch_size, per_demo_batch_size, self.embedding_size)
        obs_size = obs.size(2)
        nn_input = torch.cat([obs, demo_embedding], 2).view(-1, obs_size + self.embedding_size)
        nn_output = self.decoder(nn_input)
        return output_to_info(nn_output, self.action_space)

    @cached_property
    def distribution(self):
        return space_to_distribution(self.action_space)

    def inform_task(self, task_id, env, paths, obs):
        self.demo_obs_pool = obs

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.cast['bool'](dones)
        cnt = int(np.sum(dones))
        if cnt > 0:
            demo_ids = np.random.choice(len(self.demo_obs_pool), size=cnt, replace=True)
            # only take the last time step
            demo_obs = self.demo_obs_pool[demo_ids, -1]
            demo_obs = tensor_utils.variable(demo_obs, dtype='float', requires_grad=False)
            demo_embedding = self.encoder(demo_obs)
            if self.demo_embedding is None or self.demo_embedding.size(0) != len(dones):
                self.demo_embedding = tensor_utils.variable(demo_embedding)
            else:
                mask = tensor_utils.variable(
                    np.cast['uint8'](dones),
                    dtype='byte',
                    requires_grad=False,
                ).unsqueeze(1).expand_as(self.demo_embedding)
                self.demo_embedding[mask] = demo_embedding

    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        flat_obs = tensor_utils.variable(flat_obs, dtype='float', volatile=True)
        nn_input = torch.cat([flat_obs, self.demo_embedding], 1)
        nn_output = self.decoder(nn_input)
        info_vars = output_to_info(nn_output, self.action_space)
        agent_infos = {k: tensor_utils.to_numpy(v) for k, v in info_vars.items()}
        actions = self.distribution.sample(agent_infos)
        return actions, agent_infos
