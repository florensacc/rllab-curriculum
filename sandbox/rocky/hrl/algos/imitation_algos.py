from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.envs.base import Step
from rllab.misc import logger
from rllab.misc import ext
from rllab.envs.base import EnvSpec
from rllab.spaces.box import Box
from rllab.optimizers.minibatch_dataset import BatchDataset
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.core.network import GRUNetwork, MLP, ConvNetwork, ConvMergeNetwork
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.spaces.product import Product
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
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
        self._observation_space = Box(low=0., high=1., shape=(self.n_row, self.n_col, N_OBJECT_TYPES))

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
    learning_rate = 0.001
    batch_size = 128
    n_epochs = 100


class ImitationAlgo(RLAlgorithm):
    def __init__(self):
        pass

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
        low_policy = CategoricalMLPPolicy(
            name="low_policy",
            env_spec=EnvSpec(
                observation_space=Product(env.observation_space, high_policy.action_space),
                action_space=env.action_space,
            ),
            prob_network=ConvMergeNetwork(
                name="low_policy_network",
                input_shape=env.observation_space.components[0].shape,
                extra_input_shape=(Product(env.observation_space.components[1:]).flat_dim + SPEC.subgoal_dim,),
                output_dim=env.action_space.flat_dim,
                hidden_sizes=(10, 10),
                conv_filters=(10, 10),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('SAME', 'SAME'),
                extra_hidden_sizes=tuple(),  # (10, 10),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=tf.nn.softmax,
            )
        )

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
        action_dist_info = low_policy.dist_info_sym(low_obs, dict())
        flat_action_logli = low_policy.distribution.log_likelihood_sym(flat_action_var, action_dist_info)

        action_logli = tf.reshape(flat_action_logli, (-1, SPEC.subgoal_interval))
        sum_action_logli = tf.reduce_sum(action_logli, -1)

        subgoal_kl = high_policy.distribution.kl_sym(recog_subgoal_dist, policy_subgoal_dist)
        subgoal_logli = recog.distribution.log_likelihood_sym(recog_subgoal, recog_subgoal_dist)

        loss = tf.reduce_mean(
            - sum_action_logli + subgoal_kl
        )
        surr_loss = tf.reduce_mean(
            - sum_action_logli + subgoal_kl - tf.stop_gradient(sum_action_logli) * subgoal_logli
        )

        joint_target = JointParameterized([high_policy, low_policy, recog])

        params = joint_target.get_params(trainable=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=SPEC.learning_rate)
        train_op = optimizer.minimize(surr_loss, var_list=params)

        dataset = BatchDataset([seg_obs, seg_actions], batch_size=SPEC.batch_size)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch_id in xrange(SPEC.n_epochs):

                losses = []

                for batch_obs, batch_actions in dataset.iterate():
                    # Sample minibatch and train
                    _, loss_val = sess.run([train_op, loss], feed_dict={obs_var: batch_obs, action_var: batch_actions})
                    losses.append(loss_val)

                logger.log("Average loss: %f" % np.mean(losses))




                # jointly optimize all the parameters
                # params = high_policy.get_params(trainable=True) + low_policy.get_params(train)

                # Now we can
