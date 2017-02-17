from cached_property import cached_property

from rllab.core.serializable import Serializable
from sandbox.rocky.th.core.scope_powered import ScopePowered
from sandbox.rocky.th.distributions.utils import space_to_dist_dim, output_to_info, space_to_distribution
from sandbox.rocky.th import ops
import torch.nn.functional as F
import numpy as np
import torch


class ConvAnalogyPolicy(ScopePowered, Serializable):
    def __init__(
            self,
            env_spec,
            rates=(1, 2, 4, 8, 16, 32, 64, 128, 256),
            filter_size=2,
            residual_channels=64,
            decoder_hidden_sizes=(64, 64),
            name="policy",
    ):
        Serializable.quick_init(self, locals())
        self.name = name
        self.rates = rates
        self.residual_channels = residual_channels
        self.filter_size = filter_size
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_flat_dim = space_to_dist_dim(env_spec.action_space)
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space
        self.scope = ops.scope(self.name)
        self.demo_obs_pool = None
        self.demo_obs = None
        self.demo_embedding_computed = None
        self.demo_embedding = None
        self.past_obs = dict()
        self.ts = None
        self._gen_queues = None
        # dry run, to construct variables
        with ops.phase(ops.TEST):
            self._forward(np.zeros((1, 1, self.obs_dim)),
                          np.zeros((1, 1, self.obs_dim)))
        ScopePowered.__init__(self, scope=self.scope)

    def data_dependent_init(self, demo_paths, obs):
        self.forward(demo_paths, obs)

    def forward(self, demo_paths, obs):
        # shape: batch_size * time * dim
        demo_obs = np.asarray([p["observations"] for p in demo_paths])
        return self._forward(demo_obs, obs)

    def get_demo_embedding(self, demo_obs):
        assert demo_obs.shape[-1] == self.obs_dim
        # transform to batch_size * dim * time (as required by torch)
        demo_obs = demo_obs.transpose((0, 2, 1))
        dim = self.residual_channels
        with ops.scope("demo_encoder"):
            x = ops.wrap(demo_obs).conv1d_wn(dim=dim, size=1)
            for idx, rate in enumerate(self.rates):
                # residual pattern
                with ops.scope("res_block_{}".format(idx)):
                    x = (x.branch_out()
                         .act(F.relu)
                         .conv1d_wn(dim=dim // 2, size=1)
                         .act(F.relu)
                         .conv1d_wn(dim=dim // 2, size=self.filter_size, rate=rate)
                         .act(F.relu)
                         .conv1d_wn(dim=dim, size=1)
                         .branch_in(mode='sum'))
            demo_embedding = (x.act(F.relu)
                              .conv1d_wn(dim=dim, size=1)
                              .act(F.relu)
                              .value)
        return demo_embedding

    def get_obs_embedding(self, obs):
        assert obs.shape[-1] == self.obs_dim
        # transform to batch_size * dim * time (as required by torch)
        obs = np.asarray(obs.transpose((0, 2, 1)), order='C')
        dim = self.residual_channels
        with ops.scope("obs_encoder"):
            obs_embedding = (ops.wrap(obs)
                             .conv1d_wn(dim=dim, size=1)
                             .act(F.relu)
                             .value)
        return obs_embedding

    def get_joint_embedding(self, demo_embedding, obs_embedding):
        dim = self.residual_channels
        with ops.scope("joint_encoder"):
            joint_embedding = (ops.wrap([demo_embedding, obs_embedding], mode='concat', concat_dim=1)
                               .conv1d_wn(dim=dim, size=1)
                               .value)
        return joint_embedding

    def get_nn_output(self, joint_embedding):
        with ops.scope("decoder"):
            x = ops.wrap(joint_embedding)
            for hidden_size in self.decoder_hidden_sizes:
                x = x.act(F.relu).conv1d_wn(dim=hidden_size, size=1)
            x = x.act(F.relu).conv1d_wn(dim=self.action_flat_dim, size=1)
        nn_output = (x.value.transpose(2, 1)
                     .contiguous()
                     .view(-1, self.action_flat_dim))
        return nn_output

    def _forward(self, demo_obs, obs, demo_embedding=None):
        with ops.scope(self.scope):
            if demo_embedding is None:
                demo_embedding = self.get_demo_embedding(demo_obs)
            obs_embedding = self.get_obs_embedding(obs)
            joint_embedding = self.get_joint_embedding(
                demo_embedding, obs_embedding)

            nn_output = self.get_nn_output(joint_embedding)
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
