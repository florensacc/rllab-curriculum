from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.envs.base import Step
from rllab.misc import logger
from rllab.misc import ext
from rllab.envs.base import EnvSpec
from rllab.spaces.box import Box as TT_Box
from rllab.core.serializable import Serializable
from rllab.optimizers.minibatch_dataset import BatchDataset
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.core.network import GRUNetwork, MLP, ConvNetwork, ConvMergeNetwork
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.spaces.product import Product
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.base import StochasticPolicy
import sandbox.rocky.tf.core.layers as L
import itertools
import copy
import numpy as np
import tensorflow as tf

UP = GridWorldEnv.action_from_direction("up")
DOWN = GridWorldEnv.action_from_direction("down")
LEFT = GridWorldEnv.action_from_direction("left")
RIGHT = GridWorldEnv.action_from_direction("right")

AGENT = 0
GOAL = 1
WALL = 2
HOLE = 3
N_OBJECT_TYPES = 4


class JointParameterized(Parameterized):
    def __init__(self, components):
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        params = [param for comp in self.components for param in comp.get_params_internal(**tags)]
        # only return unique parameters
        return sorted(set(params), key=hash)


class ImageGridWorld(GridWorldEnv):
    def __init__(self, desc):
        super(ImageGridWorld, self).__init__(desc)
        self._observation_space = TT_Box(low=0., high=1., shape=(self.n_row, self.n_col, N_OBJECT_TYPES))

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self):
        super(ImageGridWorld, self).reset()
        return self.get_current_obs()

    def step(self, action):
        _, reward, done, info = super(ImageGridWorld, self).step(action)
        return Step(self.get_current_obs(), reward, done, **info)

    def get_current_obs(self):
        ret = np.zeros(self._observation_space.shape)
        ret[self.desc == 'H', HOLE] = 1
        ret[self.desc == 'W', WALL] = 1
        ret[self.desc == 'G', GOAL] = 1
        cur_x = self.state / self.n_col
        cur_y = self.state % self.n_col
        ret[cur_x, cur_y, AGENT] = 1
        return ret


def generate_demonstration_trajectory(size, start_pos, end_pos, action_map):
    sx, sy = start_pos
    ex, ey = end_pos
    actions = []
    if sx < ex:
        actions += action_map[DOWN] * (ex - sx)
    elif sx > ex:
        actions += action_map[UP] * (sx - ex)
    if sy < ey:
        actions += action_map[RIGHT] * (ey - sy)
    elif sy > ey:
        actions += action_map[LEFT] * (sy - ey)
    # Now we execute the list of actions in the environment
    base_map = [["."] * size for _ in range(size)]
    base_map[sx][sy] = 'S'
    base_map[ex][ey] = 'G'
    wrapped_env = ImageGridWorld(desc=base_map)
    env = CompoundActionSequenceEnv(wrapped_env, action_map, obs_include_history=True)
    obs = env.reset()
    observations = []
    terminal = False
    reward = 0
    for act in actions:
        next_obs, reward, terminal, _ = env.step(act)
        observations.append(obs)
        obs = next_obs
    assert terminal
    assert reward == 1
    return dict(
        observations=np.asarray(env.observation_space.flatten_n(observations)),
        actions=np.asarray(env.action_space.flatten_n(actions))
    )


class ApproximatePosterior(LayersPowered):
    """
    The approximate posterior takes in the sequence of states and actions, and predicts the hidden state based on the
    sequence. It is structured as a GRU.
    """

    def __init__(self, env_spec, subgoal_dim, subgoal_interval, feature_dim=10, hidden_dim=10):
        """
        :type env_spec: EnvSpec
        :param env_spec:
        :return:
        """
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        l_in = L.InputLayer(
            name="input",
            shape=(None, None, obs_dim + action_dim),
        )
        l_obs = L.SliceLayer(
            l_in,
            indices=slice(obs_dim),
            axis=-1,
            name="obs_input"
        )
        l_action = L.SliceLayer(
            l_in,
            indices=slice(obs_dim, obs_dim + action_dim),
            axis=-1,
            name="action_input"
        )
        # feature_dim = 10
        # hidden_dim = 10
        # subgoal_interval = 3
        feature_network = ConvMergeNetwork(
            name="feature_network",
            input_layer=L.reshape(l_obs, (-1, env_spec.observation_space.flat_dim), name="reshape_obs"),
            input_shape=env_spec.observation_space.components[0].shape,
            extra_input_shape=(Product(env_spec.observation_space.components[1:]).flat_dim,),
            output_dim=feature_dim,
            hidden_sizes=(10,),
            conv_filters=(10, 10),
            conv_filter_sizes=(3, 3),
            conv_strides=(1, 1),
            conv_pads=('SAME', 'SAME'),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.tanh,
        )
        l_reshaped_feature = L.reshape(
            feature_network.output_layer,
            shape=(-1, subgoal_interval, feature_dim),
            name="reshaped_feature"
        )
        subgoal_network = GRUNetwork(
            name="h_network",
            input_shape=(feature_dim,),
            output_dim=subgoal_dim,
            hidden_dim=hidden_dim,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.softmax,
            input_layer=L.concat([l_reshaped_feature, l_action], name="subgoal_in", axis=2),
        )
        l_subgoal_probs = L.SliceLayer(
            subgoal_network.output_layer,
            indices=subgoal_interval - 1,
            axis=1,
            name="subgoal_probs"
        )
        self.subgoal_dim = subgoal_dim
        self.l_in = l_in
        self.l_obs = l_obs
        self.l_action = l_action
        self.l_subgoal_probs = l_subgoal_probs
        LayersPowered.__init__(self, [l_subgoal_probs])

    def dist_info_sym(self, obs_var, action_var):
        assert obs_var.get_shape().ndims == 3
        assert action_var.get_shape().ndims == 3
        prob = L.get_output(self.l_subgoal_probs, {self.l_obs: obs_var, self.l_action: action_var})
        return dict(prob=prob)

    @property
    def distribution(self):
        return Categorical(self.subgoal_dim)


class BranchingCategoricalMLPPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            shared_network,
            subgoal_dim,
            bottleneck_dim,
            bottleneck_std_threshold=1e-3,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        with tf.variable_scope(name):
            l_last = shared_network.output_layer
            # map to mean and std of bottleneck
            l_bottleneck_mean = L.DenseLayer(
                l_last,
                num_units=bottleneck_dim,
                nonlinearity=tf.nn.tanh,
                name="bottleneck_mean"
            )
            l_bottleneck_std = L.DenseLayer(
                l_last,
                num_units=bottleneck_dim,
                nonlinearity=tf.exp,
                name="bottleneck_std"
            )
            l_bottleneck_std = L.OpLayer(
                l_bottleneck_std,
                op=lambda x: tf.maximum(x, bottleneck_std_threshold),
                shape_op=lambda x: x,
                name="bottleneck_std_clipped",
            )
            l_bottleneck_epsilon = L.InputLayer(shape=(None, bottleneck_dim), name="l_bottleneck_epsilon")

            l_bottleneck = L.OpLayer(
                l_bottleneck_mean, extras=[l_bottleneck_std, l_bottleneck_epsilon],
                op=lambda mean, std, epsilon: mean + std * epsilon,
                shape_op=lambda x, *args: x,
                name="bottleneck"
            )

            prob_networks = []

            for subgoal in xrange(subgoal_dim):
                prob_network = MLP(
                    input_layer=l_bottleneck,
                    output_dim=env_spec.action_space.n,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=tf.nn.softmax,
                    name="prob_network_%d" % subgoal,
                )
                prob_networks.append(prob_network)

            self.prob_networks = prob_networks
            self.l_probs = [net.output_layer for net in prob_networks]
            self.l_obs = shared_network.input_layer
            self.l_bottleneck_mean = l_bottleneck_mean
            self.l_bottleneck_std = l_bottleneck_std
            self.l_bottleneck_epsilon = l_bottleneck_epsilon

            self.bottleneck_dist = bottleneck_dist = DiagonalGaussian(dim=SPEC.bottleneck_dim)
            self.subgoal_dim = subgoal_dim
            self.dist = Categorical(env_spec.action_space.n)
            self.shared_network = shared_network

            self.bottleneck_space = bottleneck_space = Box(low=-1, high=1, shape=(bottleneck_dim,))

            super(BranchingCategoricalMLPPolicy, self).__init__(env_spec)
            LayersPowered.__init__(self, [net.output_layer for net in prob_networks])

            obs_var = self.observation_space.new_tensor_variable(
                "obs",
                extra_dims=1,
            )
            epsilon_var = bottleneck_space.new_tensor_variable(
                "bottleneck_epsilon",
                extra_dims=1,
            )

            self.f_prob = tensor_utils.compile_function(
                [obs_var, epsilon_var],
                self.dist_info_sym(obs_var, dict(bottleneck_epsilon=epsilon_var))["prob"],
            )
            self.f_bottleneck_dist_info = tensor_utils.compile_function(
                [obs_var],
                self.bottleneck_dist_info_sym(obs_var)
            )

    def bottleneck_dist_info_sym(self, obs_var):
        high_obs = obs_var[:, :self.observation_space.flat_dim - self.subgoal_dim]
        subgoals = obs_var[:, self.observation_space.flat_dim - self.subgoal_dim:]
        mean, std = L.get_output(
            [self.l_bottleneck_mean, self.l_bottleneck_std],
            {
                self.l_obs: tf.cast(high_obs, tf.float32),
            }
        )
        return dict(mean=mean, log_std=tf.log(std))

    def dist_info_sym(self, obs_var, state_info_vars):
        high_obs = obs_var[:, :self.observation_space.flat_dim - self.subgoal_dim]
        subgoals = obs_var[:, self.observation_space.flat_dim - self.subgoal_dim:]
        bottleneck_epsilon = state_info_vars["bottleneck_epsilon"]
        prob_vars = L.get_output(
            self.l_probs,
            {
                self.l_obs: tf.cast(high_obs, tf.float32),
                self.l_bottleneck_epsilon: bottleneck_epsilon,
            }
        )
        probs = tf.batch_matmul(
            tf.expand_dims(subgoals, 1),
            tf.transpose(tf.pack(prob_vars), (1, 0, 2))
        )[:, 0, :]
        return dict(prob=probs)

    def dist_info(self, obs, state_infos):
        return dict(prob=self.f_prob(obs, state_infos["bottleneck_epsilon"]))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    def get_action(self, observation):
        raise NotImplementedError
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        raise NotImplementedError
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = map(self.action_space.weighted_sample, probs)
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self.dist


class SPEC:
    size = 5
    action_map = [
        [0, 1, 1],
        [1, 3, 3],
        [2, 2, 0],
        [3, 0, 2],
    ]
    subgoal_dim = 4
    subgoal_interval = 3
    learning_rate = 0.0001
    batch_size = 128
    n_epochs = 1000
    bottleneck_dim = 10
    bottleneck_coeff = 1.


class ImitationAlgo(RLAlgorithm):
    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(SPEC, k, v)
        print(SPEC.learning_rate)
        print(SPEC.bottleneck_coeff)

    def collect_paths(self):
        logger.log("generating paths")
        all_pos = [(x, y) for x in range(SPEC.size) for y in range(SPEC.size)]
        paths = []
        for start_pos, end_pos in itertools.combinations(all_pos, 2):
            # generate a demonstration trajectory from the start to the end
            path = generate_demonstration_trajectory(SPEC.size, start_pos, end_pos, SPEC.action_map)
            paths.append(path)
        logger.log("generated")
        return paths

    def log_diagnostics(self, env, high_policy, low_policy, seg_obs, seg_actions):
        # There's a few metrics we want to look at
        # - I(a;h|s), indicating how much the low-level policy is listening to the high-level one
        # unflat_obs = env.observation_space.unflatten_n(observations)

        observations = seg_obs[:, 0, :]
        actions = seg_actions[:, 0, :]
        N = observations.shape[0]

        all_low_probs = []

        for g in xrange(SPEC.subgoal_dim):
            subgoals = np.tile(
                high_policy.action_space.flatten(g).reshape((1, -1)),
                (N, 1)
            )
            low_obs = np.concatenate([observations, subgoals], axis=-1)
            bottleneck_epsilons = np.random.normal(size=(N, SPEC.bottleneck_dim))
            low_probs = low_policy.dist_info(low_obs, dict(bottleneck_epsilon=bottleneck_epsilons))["prob"]
            all_low_probs.append(low_probs)

        all_low_probs = np.asarray(all_low_probs)
        flat_low_probs = all_low_probs.reshape((-1, low_policy.action_space.n))

        p_a_given_s = np.mean(all_low_probs, axis=0)
        h_a_given_s = np.mean(low_policy.distribution.entropy(dict(prob=p_a_given_s)))
        h_a_given_h_s = np.mean(low_policy.distribution.entropy(dict(prob=flat_low_probs)))

        mi_a_h_given_s = h_a_given_s - h_a_given_h_s

        bottleneck_dist_info = low_policy.f_bottleneck_dist_info(low_obs)
        prior_bottleneck_dist_info = dict(mean=np.zeros((N, SPEC.bottleneck_dim)),
                                          log_std=np.zeros((N, SPEC.bottleneck_dim)))
        bottleneck_kl = np.mean(low_policy.bottleneck_dist.kl(bottleneck_dist_info, prior_bottleneck_dist_info))

        # import ipdb; ipdb.set_trace()

        logger.record_tabular("I(a;h|s)", mi_a_h_given_s)
        logger.record_tabular("H(a|s)", h_a_given_s)
        logger.record_tabular("H(a|h,s)", h_a_given_h_s)
        logger.record_tabular("KL(p(z|s)||p(z))", bottleneck_kl)

    def train(self):
        paths = self.collect_paths()
        # With our particular construction, we could simply break the trajectories into segments of length 3
        observations = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        N = observations.shape[0]

        seg_obs = observations.reshape((N / 3, 3, -1))
        seg_actions = actions.reshape((N / 3, 3, -1))

        base_map = [["."] * SPEC.size for _ in range(SPEC.size)]
        base_map[0][0] = 'S'

        wrapped_env = ImageGridWorld(desc=base_map)
        env = TfEnv(CompoundActionSequenceEnv(wrapped_env, SPEC.action_map, obs_include_history=True))
        obs_dim = env.observation_space.flat_dim
        action_dim = env.action_space.flat_dim

        subgoal_space = Discrete(SPEC.subgoal_dim)
        high_policy = CategoricalMLPPolicy(
            name="high_policy",
            env_spec=EnvSpec(
                observation_space=env.observation_space,
                action_space=subgoal_space,
            ),
            prob_network=ConvMergeNetwork(
                name="high_policy_network",
                input_shape=env.observation_space.components[0].shape,
                extra_input_shape=(Product(env.observation_space.components[1:]).flat_dim,),
                output_dim=SPEC.subgoal_dim,
                hidden_sizes=(10, 10),
                conv_filters=(10, 10),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('SAME', 'SAME'),
                extra_hidden_sizes=tuple(),  # (10, 10),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=tf.nn.softmax,
            ),
        )
        low_policy = BranchingCategoricalMLPPolicy(
            name="low_policy",
            env_spec=EnvSpec(
                observation_space=Product(env.observation_space, high_policy.action_space),
                action_space=env.action_space,
            ),
            subgoal_dim=SPEC.subgoal_dim,
            shared_network=ConvMergeNetwork(
                name="low_policy_shared_network",
                input_shape=env.observation_space.components[0].shape,
                extra_input_shape=(Product(env.observation_space.components[1:]).flat_dim,),
                output_dim=32,  # env.action_space.flat_dim,
                hidden_sizes=tuple(),  # (10,),
                conv_filters=(10, 10),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('SAME', 'SAME'),
                extra_hidden_sizes=tuple(),  # (10, 10),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=tf.nn.tanh,
            ),
            hidden_sizes=(32,),
            hidden_nonlinearity=tf.nn.tanh,
            bottleneck_dim=SPEC.bottleneck_dim,
        )
        # low_policy = CategoricalMLPPolicy(
        #     name="low_policy",
        #     env_spec=EnvSpec(
        #         observation_space=Product(env.observation_space, high_policy.action_space),
        #         action_space=env.action_space,
        #     ),
        #     prob_network=ConvMergeNetwork(
        #         name="low_policy_network",
        #         input_shape=env.observation_space.components[0].shape,
        #         extra_input_shape=(Product(env.observation_space.components[1:]).flat_dim + SPEC.subgoal_dim,),
        #         output_dim=env.action_space.flat_dim,
        #         hidden_sizes=(10, 10),
        #         conv_filters=(10, 10),
        #         conv_filter_sizes=(3, 3),
        #         conv_strides=(1, 1),
        #         conv_pads=('SAME', 'SAME'),
        #         extra_hidden_sizes=tuple(),  # (10, 10),
        #         hidden_nonlinearity=tf.nn.tanh,
        #         output_nonlinearity=tf.nn.softmax,
        #     )
        # )

        recog = ApproximatePosterior(env_spec=env.spec, subgoal_dim=SPEC.subgoal_dim,
                                     subgoal_interval=SPEC.subgoal_interval)

        # There's the option for batch optimization vs. minibatch
        # Since we need to sample from the approx. posterior anyways, we'll go for minibatch
        obs_var = env.observation_space.new_tensor_variable(
            name="obs",
            extra_dims=2,
        )
        action_var = env.action_space.new_tensor_variable(
            name="action",
            extra_dims=2,
        )
        bottleneck_epsilon_var = tf.placeholder(
            dtype=tf.float32,
            shape=(None, SPEC.bottleneck_dim),
            name="bottleneck_epsilon"
        )
        # Sample h~q(h|s, a)
        # Should return the same dimension
        recog_subgoal_dist = recog.dist_info_sym(obs_var, action_var)
        recog_subgoal = recog.distribution.sample_sym(recog_subgoal_dist)
        flat_obs_var = tf.reshape(obs_var, (-1, obs_dim))
        flat_action_var = tf.reshape(action_var, (-1, action_dim))

        high_obs_var = obs_var[:, 0, :]
        policy_subgoal_dist = high_policy.dist_info_sym(high_obs_var, dict())

        # tile the subgoals to match the dimension of obs / actions
        tiled_recog_subgoal = tf.tile(
            tf.expand_dims(recog_subgoal, 1),
            [1, SPEC.subgoal_interval, 1]
        )

        flat_recog_subgoal = tf.reshape(tiled_recog_subgoal, (-1, SPEC.subgoal_dim))

        low_obs = tf.concat(1, [flat_obs_var, flat_recog_subgoal])
        action_dist_info = low_policy.dist_info_sym(low_obs, dict(bottleneck_epsilon=bottleneck_epsilon_var))
        flat_action_logli = low_policy.distribution.log_likelihood_sym(flat_action_var, action_dist_info)

        action_logli = tf.reshape(flat_action_logli, (-1, SPEC.subgoal_interval))
        sum_action_logli = tf.reduce_sum(action_logli, -1)

        subgoal_kl = high_policy.distribution.kl_sym(recog_subgoal_dist, policy_subgoal_dist)
        subgoal_logli = recog.distribution.log_likelihood_sym(recog_subgoal, recog_subgoal_dist)

        bottleneck_dist_info = low_policy.bottleneck_dist_info_sym(low_obs)
        prior_bottleneck_dist_info = dict(
            mean=tf.zeros_like(bottleneck_dist_info["mean"]),
            log_std=tf.zeros_like(bottleneck_dist_info["log_std"])
        )
        bottleneck_kl = low_policy.bottleneck_dist.kl_sym(bottleneck_dist_info, prior_bottleneck_dist_info)
        # bottleneck_kl = tf.reshape(flat_bottleneck_kl, (-1, SPEC.subgoal_interval))

        vlb = tf.reduce_mean(- sum_action_logli + subgoal_kl)
        avg_bottleneck_kl = tf.reduce_mean(bottleneck_kl)

        loss = vlb + SPEC.bottleneck_coeff * avg_bottleneck_kl
        surr_loss = loss - tf.reduce_mean(tf.stop_gradient(sum_action_logli) * subgoal_logli)

        joint_target = JointParameterized([high_policy, low_policy, recog])

        params = joint_target.get_params(trainable=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=SPEC.learning_rate)
        train_op = optimizer.minimize(surr_loss, var_list=params)

        dataset = BatchDataset([seg_obs, seg_actions], batch_size=SPEC.batch_size)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch_id in xrange(SPEC.n_epochs):

                losses = []
                vlbs = []
                bottleneck_kls = []

                for batch_obs, batch_actions in dataset.iterate():
                    # Sample minibatch and train
                    N = batch_obs.shape[0] * batch_obs.shape[1]
                    epsilons = np.random.normal(size=(N, SPEC.bottleneck_dim))
                    _, loss_val, vlb_val, bottleneck_kl_val = sess.run(
                        [train_op, loss, vlb, avg_bottleneck_kl],
                        feed_dict={obs_var: batch_obs, action_var: batch_actions, bottleneck_epsilon_var: epsilons}
                    )
                    losses.append(loss_val)
                    vlbs.append(vlb_val)
                    bottleneck_kls.append(bottleneck_kl_val)

                logger.record_tabular("AverageLoss", np.mean(losses))
                logger.record_tabular("AverageVlb", np.mean(vlbs))
                logger.record_tabular("AverageBottleneckKL", np.mean(bottleneck_kls))
                self.log_diagnostics(env, high_policy, low_policy, seg_obs, seg_actions)
                logger.dump_tabular()




                # jointly optimize all the parameters
                # params = high_policy.get_params(trainable=True) + low_policy.get_params(train)

                # Now we can
