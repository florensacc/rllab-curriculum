from cached_property import cached_property
from torch import nn

from rllab.core.serializable import Serializable
from sandbox.rocky.th import tensor_utils
from sandbox.rocky.th.core.module_powered import ModulePowered
from sandbox.rocky.th.core.modules import MLP, DilatedConvNet, LayerSpec, identity
from sandbox.rocky.th.distributions.utils import space_to_dist_dim, output_to_info, space_to_distribution
import torch.nn.functional as F
import numpy as np
import torch


class ConvAnalogyPolicy(ModulePowered, Serializable):
    def __init__(
            self,
            env_spec,
            encoder_spec,
            obs_embedding_hidden_sizes,
            decoder_spec,
            nonlinearity=F.relu,
            weight_norm=False,
    ):
        Serializable.quick_init(self, locals())
        super().__init__()
        obs_dim = env_spec.observation_space.flat_dim
        action_flat_dim = space_to_dist_dim(env_spec.action_space)
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space
        assert not encoder_spec.causal
        self.encoder = DilatedConvNet(
            in_channels=obs_dim,
            spec=encoder_spec,
            nonlinearity=nonlinearity,
            output_nonlinearity=nonlinearity,
        )
        if len(obs_embedding_hidden_sizes) > 0:
            self.obs_embedder = MLP(
                input_size=obs_dim,
                output_size=obs_embedding_hidden_sizes[-1],
                hidden_sizes=obs_embedding_hidden_sizes[:-1],
                nonlinearity=nonlinearity,
                output_nonlinearity=nonlinearity,
                weight_norm=False,
            )
            self.obs_embedding_dim = obs_embedding_hidden_sizes[-1]
        else:
            self.obs_embedder = identity
            self.obs_embedding_dim = obs_dim
        assert decoder_spec.causal
        self.decoder = DilatedConvNet(
            in_channels=self.obs_embedding_dim + encoder_spec.n_channels,
            spec=decoder_spec,
            nonlinearity=nonlinearity,
            output_nonlinearity=nonlinearity,
        )
        self.out_conv = nn.Conv1d(
            in_channels=decoder_spec.n_channels,
            out_channels=space_to_dist_dim(self.action_space),
            kernel_size=1,
        )

    def forward(self, demo_paths, obs):
        """
        :param demo_obs: observations along demo traj. Layout should be time * batch_size * dim
        :param obs: observations. Layout should be batch_size * dim
        :return:
        """
        demo_batch_size = len(demo_paths)
        per_demo_batch_size = obs.size(1)
        # only take the final states
        demo_obs = np.asarray([p["observations"] for p in demo_paths])
        # transform to N*C_in*L_in
        demo_obs = demo_obs.transpose((0, 2, 1))
        demo_obs = tensor_utils.variable(demo_obs, dtype='float', requires_grad=False)
        obs_size = obs.size(2)
        demo_embedding = self.encoder(demo_obs)

        obs = obs.view(-1, obs_size)
        obs_embedding = self.obs_embedder(obs) \
            .view(demo_batch_size, per_demo_batch_size, self.obs_embedding_dim) \
            .transpose(1, 2)  # transform to N*embedding_dim*Time

        joint_embedding = torch.cat([obs_embedding, demo_embedding], 1)
        decoder_output = self.decoder(joint_embedding)
        nn_output = self.out_conv(decoder_output).view(-1, space_to_dist_dim(self.action_space))

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
