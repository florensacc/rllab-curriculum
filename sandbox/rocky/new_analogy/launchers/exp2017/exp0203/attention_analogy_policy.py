from cached_property import cached_property

from rllab.core.serializable import Serializable
from sandbox.rocky.th.core.scope_powered import ScopePowered
from sandbox.rocky.th.distributions.utils import space_to_dist_dim, output_to_info, space_to_distribution
from sandbox.rocky.th import ops
import torch.nn.functional as F
import numpy as np
import torch


class AttentionAnalogyPolicy(ScopePowered, Serializable):
    def __init__(
            self,
            env_spec,
            # rates=(1, 2, 4, 8, 16, 32, 64, 128, 256),
            # filter_size=2,
            # residual_channels=64,
            # decoder_hidden_sizes=(64, 64),
            name="policy",
    ):
        Serializable.quick_init(self, locals())
        self.name = name
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_flat_dim = space_to_dist_dim(env_spec.action_space)
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space
        self.scope = ops.scope(self.name)
        ScopePowered.__init__(self, scope=self.scope)

    def data_dependent_init(self, demo_paths, obs):
        # dry run, to construct variables
        with ops.phase(ops.TEST):
            self.forward(demo_paths, obs)

    def forward(self, demo_paths, obs):
        # shape: batch_size * time * dim
        demo_obs = np.asarray([p["observations"] for p in demo_paths])
        return self._forward(demo_obs, obs)

    def _forward(self, demo_obs, obs):
        with ops.scope(self.scope):
            x = ops.wrap(demo_obs)
            x = x.blstm(dim=64)

        import ipdb; ipdb.set_trace()


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
            for idx, done in enumerate(dones):
                if done:
                    self.past_obs[idx] = []
            demo_ids = np.random.choice(
                len(self.demo_obs_pool), size=cnt, replace=True)
            demo_obs = self.demo_obs_pool[demo_ids]
            if self.demo_obs is None or self.demo_obs.shape[0] != len(dones):
                self.demo_obs = demo_obs
                self.demo_embedding_computed = np.cast[
                    'bool']([False] * len(dones))
                embedding_shape = (
                    demo_obs.shape[0], self.residual_channels, demo_obs.shape[1])
                self.demo_embedding = np.empty(embedding_shape)
                self.ts = np.zeros(cnt, dtype=np.int)
                self._gen_queues = dict()
            else:
                self.demo_obs[dones] = demo_obs
                self.demo_embedding_computed[dones] = False
                self.ts[dones] = 0
                for queue in self._gen_queues.values():
                    for item in queue:
                        item[dones] = 0.

    def fast_reset(self, dones=None):
        self.reset()

    def fast_get_action(self, observation):
        actions, outputs = self.fast_get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def fast_get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)

        with ops.scope(self.scope):
            demo_to_compute = np.logical_not(self.demo_embedding_computed)
            if np.any(demo_to_compute):
                demo_embedding = self.get_demo_embedding(
                    self.demo_obs[demo_to_compute])
                self.demo_embedding[
                    demo_to_compute] = ops.to_numpy(demo_embedding)
                self.demo_embedding_computed[:] = True

            # remember the data flow:
            # embed demo -> embed obs -> combine to joint embedding -> get causal conv output
            # we only need the demo embedding at this particular time step
            # flat_obs should have shape batch_size * obs_dim
            # demo_embedding is of shape batch_size * embedding_dim * t
            obs_embedding = self.get_obs_embedding(np.expand_dims(flat_obs, 1))

            batch_size = len(observations)

            demo_embedding = np.zeros((batch_size, self.residual_channels))
            fillable_mask = self.ts < self.demo_embedding.shape[-1]
            demo_embedding[fillable_mask] = self.demo_embedding[
                                            np.arange(np.sum(fillable_mask)), :, self.ts[fillable_mask]]

            joint_embedding = self.get_joint_embedding(
                np.expand_dims(demo_embedding, 2), obs_embedding)

            with ops.registering("conv1d_wn", ops.mk_conv1d_generation(ops.conv1d_wn, "conv1d_wn", self._gen_queues)):
                nn_output = self.get_nn_output(joint_embedding)

        self.ts += 1
        info_vars = output_to_info(nn_output, self.action_space)
        agent_infos = {k: ops.to_numpy(
            v) for k, v in info_vars.items()}
        actions = self.distribution.sample(agent_infos)
        return actions, agent_infos

    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        for idx, obs in enumerate(flat_obs):
            self.past_obs[idx].append(obs)
        max_len = max([len(self.past_obs[idx])
                       for idx in range(len(flat_obs))])
        past_obs_arr = np.asarray([
                                      np.pad(self.past_obs[idx], [[0, max_len - len(self.past_obs[idx])], [0, 0]],
                                             mode='constant')
                                      for idx in range(len(flat_obs))
                                      ])
        batch_size, t, _ = past_obs_arr.shape

        demo_to_compute = np.logical_not(self.demo_embedding_computed)
        if np.any(demo_to_compute):
            with ops.scope(self.scope):
                demo_embedding = self.get_demo_embedding(
                    self.demo_obs[demo_to_compute])
                self.demo_embedding[
                    demo_to_compute] = ops.to_numpy(demo_embedding)
                self.demo_embedding_computed[:] = True

        # After we get the embedding, we only need as long as the observations
        demo_embedding = self.demo_embedding[:, :, :t]

        info_vars = self._forward(None, past_obs_arr, demo_embedding=ops.as_variable(
            demo_embedding))

        agent_infos = dict()
        for k, v in info_vars.items():
            agent_infos[k] = ops.to_numpy(
                v.view(batch_size, t, -1)[:, -1, :])

        actions = self.distribution.sample(agent_infos)

        return actions, agent_infos
