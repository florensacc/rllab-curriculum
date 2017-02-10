from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import logger
import tensorflow as tf
from sandbox.rocky.tf.core.network import MLP
import sandbox.rocky.tf.core.layers as L
import numpy as np

TINY = 1e-8


class GCLCostLearnerFixing(object):
    """
    Implement guided cost learning.
    """

    def __init__(
            self,
            env_spec,
            demo_paths,
            policy,
            n_epochs=2,
            batch_size=128,
            learning_rate=1e-3,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            include_actions=True,
            demo_mixture_ratio=0.,
            cost_form="linear",
    ):
        """
        :param demo_mixture_ratio: the ratio of demonstration trajectories to be mixed in the contrastive samples (
        i.e. samples flagged to be from the policy distribution). This could smooth the cost a bit
        :param cost_form: Form of the cost function. Can be one of the following:
            - "linear": an unbounded linear function of the features
            - "sum_square": sum of squares of feature terms
            - "square": square of linear combination of features
            - "softplus": use the softplus nonlinearity to ensure nonnegative costs
        """
        self.env_spec = env_spec
        self.demo_paths = demo_paths
        self.policy = policy
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        l_obs = L.InputLayer(
            shape=(None, obs_dim),
        )

        l_action = L.InputLayer(
            shape=(None, action_dim),
        )

        assert len(hidden_sizes) >= 1

        if include_actions:
            network = MLP(
                name="disc",
                input_shape=(obs_dim + action_dim,),
                input_layer=L.concat([l_obs, l_action], axis=1),
                output_dim=hidden_sizes[-1],
                hidden_sizes=hidden_sizes[:-1],
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=hidden_nonlinearity,
            )
        else:
            network = MLP(
                name="disc",
                input_shape=(obs_dim,),
                input_layer=l_obs,
                output_dim=hidden_sizes[-1],
                hidden_sizes=hidden_sizes[:-1],
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=hidden_nonlinearity,
            )

        if cost_form == "linear":
            l_cost = L.DenseLayer(
                network.output_layer,
                num_units=1,
                nonlinearity=None
            )
        elif cost_form == "sum_square":
            l_cost = L.OpLayer(
                network.output_layer,
                op=lambda x: tf.reduce_sum(tf.square(x), reduction_indices=-1, keep_dims=True),
                shape_op=lambda shape: shape[:-1] + (1,),
            )
        elif cost_form == "square":
            l_cost = L.DenseLayer(
                network.output_layer,
                num_units=1,
                nonlinearity=tf.square,
            )
        elif cost_form == "softplus":
            l_cost = L.DenseLayer(
                network.output_layer,
                num_units=1,
                nonlinearity=tf.nn.softplus,
            )
        else:
            raise NotImplementedError

        # l_cost =

        self.network = network
        self.l_obs = l_obs
        self.l_action = l_action
        self.l_cost = l_cost
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.demo_mixture_ratio = demo_mixture_ratio
        self.init_opt()

    def init_opt(self):
        demo_obs_var = self.observation_space.new_tensor_variable(name="demo_obs", extra_dims=2)
        demo_action_var = self.action_space.new_tensor_variable(name="demo_action", extra_dims=2)
        demo_valid_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="demo_valid")
        pol_obs_var = self.observation_space.new_tensor_variable(name="pol_obs", extra_dims=2)
        pol_action_var = self.action_space.new_tensor_variable(name="pol_action", extra_dims=2)
        pol_valid_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="pol_valid")

        demo_N = tf.shape(demo_obs_var)[0]
        demo_T = tf.shape(demo_obs_var)[1]
        pol_N = tf.shape(pol_obs_var)[0]
        pol_T = tf.shape(pol_obs_var)[1]

        dist = self.policy.distribution

        pol_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None, None] + list(shape), name='pol_dist_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        pol_dist_info_vars_list = [pol_dist_info_vars[k] for k in dist.dist_info_keys]

        # log p(at|st)
        pol_logli_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="pol_logli")

        flat_demo_obs_var = tf.reshape(demo_obs_var, tf.pack([demo_N * demo_T, -1]))
        flat_demo_action_var = tf.reshape(demo_action_var, tf.pack([demo_N * demo_T, -1]))
        flat_pol_obs_var = tf.reshape(pol_obs_var, tf.pack([pol_N * pol_T, -1]))
        flat_pol_action_var = tf.reshape(pol_action_var, tf.pack([pol_N * pol_T, -1]))

        flat_demo_cost_var = L.get_output(self.l_cost, {self.l_obs: flat_demo_obs_var, self.l_action:
            flat_demo_action_var})[:, 0]
        demo_cost_var = tf.reshape(flat_demo_cost_var, tf.pack([demo_N, demo_T]))
        flat_pol_cost_var = L.get_output(self.l_cost, {self.l_obs: flat_pol_obs_var, self.l_action:
            flat_pol_action_var})[:, 0]
        pol_cost_var = tf.reshape(flat_pol_cost_var, tf.pack([pol_N, pol_T]))

        avg_demo_cost = tf.reduce_sum(demo_cost_var * demo_valid_var) / tf.cast(demo_N, tf.float32)

        partition_logits = - pol_cost_var - pol_logli_var
        # sum across the temporal axis
        traj_partition_logits = tf.reduce_sum(partition_logits * pol_valid_var, reduction_indices=-1)

        est_log_partition = tf.reduce_logsumexp(traj_partition_logits) - tf.log(tf.cast(pol_N, tf.float32))

        loss = avg_demo_cost + est_log_partition

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            loss, var_list=L.get_all_params(self.l_cost, trainable=True))

        self.f_train = tensor_utils.compile_function(
            inputs=[demo_obs_var, demo_action_var, demo_valid_var, pol_obs_var, pol_action_var, pol_logli_var,
                    pol_valid_var],
            outputs=train_op,
        )
        self.f_loss = tensor_utils.compile_function(
            inputs=[demo_obs_var, demo_action_var, demo_valid_var, pol_obs_var, pol_action_var, pol_logli_var,
                    pol_valid_var],
            outputs=loss,
        )
        self.f_reward = tensor_utils.compile_function(
            inputs=[pol_obs_var, pol_action_var, pol_valid_var],
            outputs=-pol_cost_var,
        )
        self.f_logli = tensor_utils.compile_function(
            inputs=[pol_action_var] + pol_dist_info_vars_list + [pol_valid_var],
            outputs=dist.log_likelihood_sym(pol_action_var, pol_dist_info_vars),
        )

        max_path_length = np.max([len(p["rewards"]) for p in self.demo_paths])
        demo_observations = [p["observations"] for p in self.demo_paths]
        demo_observations = tensor_utils.pad_tensor_n(demo_observations, max_path_length)
        demo_actions = [p["actions"] for p in self.demo_paths]
        demo_actions = tensor_utils.pad_tensor_n(demo_actions, max_path_length)
        demo_valids = [np.ones(len(path["observations"])) for path in self.demo_paths]
        demo_valids = tensor_utils.pad_tensor_n(demo_valids, max_path_length)

        demo_data = [demo_observations, demo_actions, demo_valids]
        demo_N = len(demo_observations)

        self.demo_observations = demo_observations
        self.demo_actions = demo_actions
        self.demo_valids = demo_valids
        self.demo_data = demo_data
        self.demo_N = demo_N

    def fit(self, paths):
        # Refit the discriminator on the new data
        # demo_observations = self.demo_observations
        # demo_actions = self.demo_actions
        # demo_labels = self.demo_labels
        demo_data = self.demo_data
        demo_N = self.demo_N

        max_path_length = np.max([len(p["rewards"]) for p in paths])
        pol_observations = [p["observations"] for p in paths]
        pol_observations = tensor_utils.pad_tensor_n(pol_observations, max_path_length)
        pol_actions = [p["actions"] for p in paths]
        pol_actions = tensor_utils.pad_tensor_n(pol_actions, max_path_length)
        pol_valids = [np.ones(len(path["observations"])) for path in paths]
        pol_valids = tensor_utils.pad_tensor_n(pol_valids, max_path_length)

        # pol_data = [pol_observations, pol_actions, pol_valids]
        # pol_N = len(pol_observations)

        # pol_observations = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        # pol_actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        # pol_agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        pol_agent_infos = [path["agent_infos"] for path in paths]
        pol_agent_infos = tensor_utils.stack_tensor_dict_list(
            [tensor_utils.pad_tensor_dict(p, max_path_length) for p in pol_agent_infos]
        )

        # import ipdb;
        # ipdb.set_trace()

        pol_dist_info_list = [pol_agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        pol_logli = self.f_logli(pol_actions, *pol_dist_info_list, pol_valids)
        #
        pol_N = len(pol_observations)
        # If this is changed, need to change how the labeling is handled below!
        pol_data = [pol_observations, pol_actions, pol_logli, pol_valids]

        # import ipdb;
        # ipdb.set_trace()

        assert self.demo_mixture_ratio == 0
        # # sample some data from the demonstration
        # demo_mix_N = int((self.demo_mixture_ratio / (1 - self.demo_mixture_ratio)) * pol_N)
        # demo_mix_ids = np.random.choice(np.arange(self.demo_N), size=demo_mix_N, replace=True)
        # demo_mix_data = [x[demo_mix_ids] for x in self.demo_data]
        #
        # # Estimate partition function
        # # Z ~ 1/|D|
        # # for these trajectories, use the reward as the approximate logli
        # demo_mix_logli = self.f_reward(*demo_mix_data)
        # demo_mix_data.append(demo_mix_logli)
        # pol_data = [np.concatenate([x, y], axis=0) for x, y in zip(pol_data, demo_mix_data)]
        # pol_N += demo_mix_N
        # import ipdb;
        # ipdb.set_trace()


        losses = []
        for batch_idx in range(0, pol_N, self.batch_size):
            demo_batch = [x[(batch_idx % demo_N):(batch_idx % demo_N) + self.batch_size] for x in demo_data]
            pol_batch = [x[batch_idx:batch_idx + self.batch_size] for x in pol_data]
            losses.append(self.f_loss(*(demo_batch + pol_batch)))
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
                self.f_train(*(demo_batch + pol_batch))

        losses = []
        for batch_idx in range(0, pol_N, self.batch_size):
            demo_batch = [x[(batch_idx % demo_N):(batch_idx % demo_N) + self.batch_size] for x in demo_data]
            pol_batch = [x[batch_idx:batch_idx + self.batch_size] for x in pol_data]
            losses.append(self.f_loss(*(demo_batch + pol_batch)))
        loss_after = np.mean(losses)

        logger.record_tabular('DiscLossBefore', loss_before)
        logger.record_tabular('DiscLossAfter', loss_after)

    def batch_predict(self, paths):

        max_path_length = np.max([len(p["rewards"]) for p in paths])
        pol_observations = [p["observations"] for p in paths]
        pol_observations = tensor_utils.pad_tensor_n(pol_observations, max_path_length)
        pol_actions = [p["actions"] for p in paths]
        pol_actions = tensor_utils.pad_tensor_n(pol_actions, max_path_length)
        pol_valids = [np.ones(len(path["observations"])) for path in paths]
        pol_valids = tensor_utils.pad_tensor_n(pol_valids, max_path_length)

        rewards = self.f_reward(pol_observations, pol_actions, pol_valids)

        ret = []
        for idx, path in enumerate(paths):
            ret.append(rewards[idx, :len(path["rewards"])])

        logger.record_tabular_misc_stat('DiscReward', rewards, placement='front')

        return ret
