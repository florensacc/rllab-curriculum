from cached_property import cached_property
from torch import nn

from sandbox.rocky.th import tensor_utils
from sandbox.rocky.th.core.modules import MLP
from sandbox.rocky.th.distributions.utils import space_to_dist_dim, output_to_info, space_to_distribution
from torch.autograd import Variable
import numpy as np
import torch


def embed_task_id(task_id):
    return tuple([ord(x) - ord('a') for x in task_id])


class GroundTruthAnalogyPolicy(nn.Module):
    def __init__(self, env_spec, embedding_dim=32, hidden_sizes=(128, 128)):
        super().__init__()
        obs_dim = env_spec.observation_space.flat_dim
        action_flat_dim = space_to_dist_dim(env_spec.action_space)
        self.source_encoder = nn.Embedding(
            num_embeddings=5,
            embedding_dim=embedding_dim,
        )
        self.target_encoder = nn.Embedding(
            num_embeddings=5,
            embedding_dim=embedding_dim,
        )
        self.decoder = MLP(
            input_size=obs_dim + embedding_dim * 2,
            output_size=action_flat_dim,
            hidden_sizes=hidden_sizes,
        )
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space
        self.hidden_sizes = hidden_sizes
        self.task_id = None
        self.demo_embedding = None

    def forward(self, demo_paths, obs):
        """
        :param demo_obs: observations along demo traj. Layout should be time * batch_size * dim
        :param obs: observations. Layout should be batch_size * dim
        :return:
        """
        demo_task_ids = [p["task_id"] for p in demo_paths]
        demo_embedding = self.get_embedding(demo_task_ids)
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

    def get_embedding(self, demo_task_ids):
        embedded_ids = list(map(embed_task_id, demo_task_ids))
        source_ids, target_ids = zip(*embedded_ids)
        source_ids = Variable(torch.from_numpy(np.asarray(source_ids)).long())
        target_ids = Variable(torch.from_numpy(np.asarray(target_ids)).long())
        if tensor_utils.is_cuda():
            source_ids = source_ids.cuda()
            target_ids = target_ids.cuda()
        source_embedding = self.source_encoder(source_ids)
        target_embedding = self.target_encoder(target_ids)
        return torch.cat([source_embedding, target_embedding], 1)

    @cached_property
    def distribution(self):
        return space_to_distribution(self.action_space)

    def inform_task(self, task_id, env, paths, obs):
        demo_embedding = self.get_embedding([task_id]).data.float()
        self.demo_embedding = Variable(demo_embedding)

    def reset(self, dones=None):
        pass

    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        flat_obs = torch.from_numpy(flat_obs).float()
        demo_embedding = self.demo_embedding
        if tensor_utils.is_cuda():
            flat_obs = flat_obs.cuda()
            demo_embedding = demo_embedding.cuda()
        flat_obs = Variable(flat_obs, volatile=True)
        batch_size = flat_obs.size(0)
        embedding_size = demo_embedding.size(1)
        demo_embedding = demo_embedding.expand(batch_size, embedding_size)
        nn_input = torch.cat([flat_obs, demo_embedding], 1)
        nn_output = self.decoder(nn_input)
        info_vars = output_to_info(nn_output, self.action_space)
        agent_infos = {k: tensor_utils.to_numpy(v) for k, v in info_vars.items()}
        actions = self.distribution.sample(agent_infos)
        return actions, agent_infos
