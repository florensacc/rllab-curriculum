from collections import deque

from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import logger
import tensorflow as tf
from sandbox.rocky.tf.core.network import MLP
import sandbox.rocky.tf.core.layers as L
import numpy as np

TINY = 1e-8


class MLPDiscriminator(object):
    def __init__(
            self,
            env_spec,
            demo_paths,
            n_epochs=2,
            batch_size=128,
            learning_rate=1e-3,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            include_actions=True,
            zero_expert_discr=False,
            sink_zero_expert_discr=True,
            use_sink_rewards=True,
            discount=0.99,
            demo_mixture_ratio=0.,
    ):
        """
        :param zero_expert_discr: Controls whether the rewards for the expert trajectories should go to 0 (True) or
        infinity (False) when the policy can mimic the expert well. zero_expert_discr=True corresponds to what happens
        when we follow the exact formulation of GAIL, and zero_expert_discr=False corresponds to the infinity
        formulation, which might encourage the policy to mimic the expert exactly (rather than getting a weaker and
        weaker signal as it does a better and better job)
        :param sink_zero_expert_discr: Similar to zero_expert_discr, but controlling the calculation of the sink reward
        :param use_sink_rewards: whether to use sink rewards padded at the end of trajectories
        :param demo_mixture_ratio: the ratio of demonstration trajectories to be mixed in the contrastive samples (
        i.e. samples flagged to be from the policy distribution). This could smooth the discriminator a bit
        """
        self.env_spec = env_spec
        self.demo_paths = demo_paths

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        l_obs = L.InputLayer(
            shape=(None, obs_dim),
        )

        l_action = L.InputLayer(
            shape=(None, action_dim),
        )

        if include_actions:
            network = MLP(
                name="disc",
                input_shape=(obs_dim + action_dim,),
                input_layer=L.concat([l_obs, l_action], axis=1),
                output_dim=1,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
            )
        else:
            network = MLP(
                name="disc",
                input_shape=(obs_dim,),
                input_layer=l_obs,
                output_dim=1,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
            )

        self.network = network
        self.l_obs = l_obs
        self.l_action = l_action
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.zero_expert_discr = zero_expert_discr
        self.last_sink_rewards = deque(maxlen=5)
        self.sink_reward = 0
        self.sink_zero_expert_discr = sink_zero_expert_discr
        self.use_sink_rewards = use_sink_rewards
        self.max_path_length = 0
        self.discount = discount
        self.demo_mixture_ratio = demo_mixture_ratio
        self.init_opt()

    def init_opt(self):
        obs_var = self.l_obs.input_var
        action_var = self.l_action.input_var
        y_var = tf.placeholder(dtype=tf.float32, shape=(None,), name="y")
        logits = L.get_output(self.network.output_layer)[:, 0]

        if self.zero_expert_discr:
            reward_var = -tf.nn.softplus(-logits)
        else:
            reward_var = tf.nn.softplus(logits)

        cross_ent_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y_var))

        loss = cross_ent_loss  # - 0.01 * ent

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            loss, var_list=self.network.get_params(trainable=True))

        self.f_train = tensor_utils.compile_function(
            inputs=[obs_var, action_var, y_var],
            outputs=train_op,
        )
        self.f_loss = tensor_utils.compile_function(
            inputs=[obs_var, action_var, y_var],
            outputs=loss,
        )
        self.f_reward = tensor_utils.compile_function(
            inputs=[obs_var, action_var],
            outputs=reward_var,
        )

        self.demo_observations = tensor_utils.concat_tensor_list([p["observations"] for p in self.demo_paths])
        self.demo_actions = tensor_utils.concat_tensor_list([p["actions"] for p in self.demo_paths])
        self.demo_labels = np.ones((len(self.demo_observations),))
        self.demo_data = [self.demo_observations, self.demo_actions, self.demo_labels]
        self.demo_lens = np.asarray([len(p["rewards"]) for p in self.demo_paths])
        self.demo_N = len(self.demo_observations)

    def fit(self, paths):
        # Refit the discriminator on the new data
        # demo_observations = self.demo_observations
        # demo_actions = self.demo_actions
        # demo_labels = self.demo_labels
        demo_data = self.demo_data
        demo_N = self.demo_N

        pol_observations = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        pol_actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        pol_labels = np.zeros((len(pol_observations),))

        pol_N = len(pol_observations)
        # If this is changed, need to change how the labeling is handled below!
        pol_data = [pol_observations, pol_actions, pol_labels]

        if self.demo_mixture_ratio > 0:
            # sample some data from the demonstration
            demo_mix_N = int((self.demo_mixture_ratio / (1 - self.demo_mixture_ratio)) * pol_N)
            demo_mix_ids = np.random.choice(np.arange(self.demo_N), size=demo_mix_N, replace=True)
            demo_mix_data = [x[demo_mix_ids] for x in self.demo_data]
            # set the labels to 0
            demo_mix_data[-1] = np.zeros((demo_mix_N,))
            pol_data = [np.concatenate([x, y], axis=0) for x, y in zip(pol_data, demo_mix_data)]
            pol_N += demo_mix_N

        # Form the Bayes-optimal estimator for the reward of the sink state

        demo_lens = self.demo_lens
        pol_lens = np.asarray([len(p["rewards"]) for p in paths])
        max_path_length = max(np.max(demo_lens), np.max(pol_lens))

        p_demo_sinks = np.mean(max_path_length - demo_lens) / max_path_length + TINY
        p_pol_sinks = np.mean(max_path_length - pol_lens) / max_path_length + TINY
        if self.demo_mixture_ratio > 0:
            p_pol_sinks = p_pol_sinks * (1 - self.demo_mixture_ratio) + p_demo_sinks * self.demo_mixture_ratio

        if self.sink_zero_expert_discr:
            sink_reward = np.log(p_demo_sinks) - np.log(p_demo_sinks + p_pol_sinks)
        else:
            if p_demo_sinks < TINY:
                sink_reward = 0
            else:
                sink_reward = np.log(p_demo_sinks + p_pol_sinks) - np.log(p_pol_sinks)
        self.last_sink_rewards.append(sink_reward)
        self.sink_reward = np.mean(self.last_sink_rewards)
        self.max_path_length = max_path_length

        losses = []
        for batch_idx in range(0, pol_N, self.batch_size):
            demo_batch = [x[(batch_idx % demo_N):(batch_idx % demo_N) + self.batch_size] for x in demo_data]
            pol_batch = [x[batch_idx:batch_idx + self.batch_size] for x in pol_data]
            joint_batch = [np.concatenate([x, y], axis=0) for x, y in zip(demo_batch, pol_batch)]
            losses.append(self.f_loss(*joint_batch))
        loss_before = np.mean(losses)

        # one epoch is as large as the minimum size of policy / demo paths
        for _ in range(self.n_epochs):
            # shuffling all data
            demo_ids = np.arange(demo_N)
            np.random.shuffle(demo_ids)
            demo_data = [x[demo_ids] for x in demo_data]
            pol_ids = np.arange(pol_N)
            np.random.shuffle(pol_ids)
            pol_data = [x[pol_ids] for x in pol_data]

            for batch_idx in range(0, pol_N, self.batch_size):
                # take samples from each sides
                demo_batch = [x[(batch_idx % demo_N):(batch_idx % demo_N) + self.batch_size] for x in demo_data]
                pol_batch = [x[batch_idx:batch_idx + self.batch_size] for x in pol_data]
                joint_batch = [np.concatenate([x, y], axis=0) for x, y in zip(demo_batch, pol_batch)]
                self.f_train(*joint_batch)

        losses = []
        for batch_idx in range(0, pol_N, self.batch_size):
            demo_batch = [x[(batch_idx % demo_N):(batch_idx % demo_N) + self.batch_size] for x in demo_data]
            pol_batch = [x[batch_idx:batch_idx + self.batch_size] for x in pol_data]
            joint_batch = [np.concatenate([x, y], axis=0) for x, y in zip(demo_batch, pol_batch)]
            losses.append(self.f_loss(*joint_batch))
        loss_after = np.mean(losses)

        # loss_after = self.f_loss(*joint_data)

        logger.record_tabular('DiscLossBefore', loss_before)
        logger.record_tabular('DiscLossAfter', loss_after)
        logger.record_tabular('SinkReward', self.sink_reward)

    def batch_predict(self, paths):
        pol_observations = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        pol_actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        rewards = self.f_reward(pol_observations, pol_actions)
        start_idx = 0
        ret = []
        for path in paths:
            path_rewards = rewards[start_idx:start_idx + len(path["rewards"])]
            if self.use_sink_rewards and len(path_rewards) < self.max_path_length:
                n_pad = (self.max_path_length - len(path_rewards))
                if 1 - self.discount < TINY:
                    path_rewards[-1] += self.sink_reward * n_pad
                else:
                    path_rewards[-1] += self.sink_reward * self.discount * (1 - self.discount ** n_pad) / (
                        1 - self.discount)
            ret.append(path_rewards)
            start_idx += len(path["rewards"])

        logger.record_tabular_misc_stat('DiscReward', rewards, placement='front')

        return ret
