from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import OrderedDict

from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from rllab.misc import ext
from sandbox.rocky.tf.core.parameterized import JointParameterized
from rllab.envs.grid_world_env import GridWorldEnv
from sandbox.rocky.hrl_imitation.envs.image_grid_world import ImageGridWorld
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from sandbox.rocky.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.optimizers.minibatch_dataset import BatchDataset
from sandbox.rocky.tf.envs.base import TfEnv
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.product import Product
from sandbox.rocky.tf.core.network import ConvMergeNetwork
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.core.network import GRUNetwork
from rllab.envs.base import EnvSpec
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.uniform_control_policy import UniformControlPolicy
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.spaces.box import Box
from rllab.core.serializable import Serializable
from rllab.misc import special
from sandbox.rocky.hrl_imitation.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.hrl_imitation.envs.dummy_vec_env import DummyVecEnv
import itertools


class SeqGridPolicyModule(object):
    """
    low-level policy receives partial observation. stochastic bottleneck
    """

    def __init__(self, low_policy_obs='full'):
        self.low_policy_obs = low_policy_obs

    def new_high_policy(self, env_spec, subgoal_dim):
        subgoal_space = Discrete(subgoal_dim)
        return CategoricalMLPPolicy(
            name="high_policy",
            env_spec=EnvSpec(
                observation_space=env_spec.observation_space,
                action_space=subgoal_space,
            ),
            prob_network=ConvMergeNetwork(
                name="high_policy_network",
                input_shape=env_spec.observation_space.components[0].shape,
                extra_input_shape=(Product(env_spec.observation_space.components[1:]).flat_dim,),
                output_dim=subgoal_dim,
                hidden_sizes=(100, 100),
                conv_filters=(10, 10),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('SAME', 'SAME'),
                extra_hidden_sizes=(100,),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=tf.nn.softmax,
            ),
        )

    def new_alt_high_policy(self, env_spec, subgoal_dim):
        subgoal_space = Discrete(subgoal_dim)
        return UniformControlPolicy(
            env_spec=EnvSpec(
                observation_space=env_spec.observation_space,
                action_space=subgoal_space,
            ),
        )

    def new_low_policy(self, env_spec, subgoal_dim, bottleneck_dim):
        subgoal_space = Discrete(subgoal_dim)
        if self.low_policy_obs == 'full':
            return BranchingCategoricalMLPPolicy(
                name="low_policy",
                env_spec=EnvSpec(
                    observation_space=Product(env_spec.observation_space, subgoal_space),
                    action_space=env_spec.action_space,
                ),
                shared_network=ConvMergeNetwork(
                    name="low_policy_shared_network",
                    input_shape=env_spec.observation_space.components[0].shape,
                    extra_input_shape=(Product(env_spec.observation_space.components[1:]).flat_dim,),
                    output_dim=100,
                    hidden_sizes=(100,),
                    conv_filters=(10, 10),
                    conv_filter_sizes=(3, 3),
                    conv_strides=(1, 1),
                    conv_pads=('SAME', 'SAME'),
                    extra_hidden_sizes=(100,),
                    hidden_nonlinearity=tf.nn.tanh,
                    output_nonlinearity=tf.nn.tanh,
                ),
                subgoal_dim=subgoal_dim,
                hidden_sizes=(100,),
                hidden_nonlinearity=tf.nn.tanh,
                bottleneck_dim=bottleneck_dim,
            )
        elif self.low_policy_obs == 'partial':
            return IgnorantBranchingCategoricalMLPPolicy(
                name="low_policy",
                env_spec=EnvSpec(
                    observation_space=Product(env_spec.observation_space, subgoal_space),
                    action_space=env_spec.action_space,
                ),
                subgoal_dim=subgoal_dim,
                hidden_sizes=(100,),
                hidden_nonlinearity=tf.nn.tanh,
                bottleneck_dim=bottleneck_dim,
            )
        else:
            raise NotImplementedError


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

            # l_bottleneck = L.DenseLayer(
            #     l_last,
            #     num_units=bottleneck_dim,
            #     nonlinearity=tf.nn.tanh,
            #     name="bottleneck"
            # )

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
            self.l_obs = [x for x in L.get_all_layers(shared_network.input_layer) if isinstance(x, L.InputLayer)][0]
            self.l_bottleneck = l_bottleneck
            self.l_bottleneck_mean = l_bottleneck_mean
            self.l_bottleneck_std = l_bottleneck_std
            self.l_bottleneck_epsilon = l_bottleneck_epsilon
            self.bottleneck_dim = bottleneck_dim

            self.bottleneck_dist = bottleneck_dist = DiagonalGaussian(dim=bottleneck_dim)
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

            self.f_bottleneck = tensor_utils.compile_function(
                [self.l_obs.input_var, self.l_bottleneck_epsilon.input_var],
                self.bottleneck_sym(self.l_obs.input_var, dict(bottleneck_epsilon=self.l_bottleneck_epsilon.input_var)),
            )
            self.f_prob = tensor_utils.compile_function(
                [obs_var, self.l_bottleneck_epsilon.input_var],
                self.dist_info_sym(obs_var, dict(bottleneck_epsilon=self.l_bottleneck_epsilon.input_var))["prob"],
            )
            self.f_bottleneck_dist_info = tensor_utils.compile_function(
                [self.l_obs.input_var],
                self.bottleneck_dist_info_sym(self.l_obs.input_var)
            )

    def bottleneck_dist_info_sym(self, high_obs_var):
        mean, std = L.get_output(
            [self.l_bottleneck_mean, self.l_bottleneck_std],
            {
                self.l_obs: high_obs_var
            }
        )
        return dict(mean=mean, log_std=tf.log(std))

    def bottleneck_sym(self, high_obs_var, state_info_vars):
        return L.get_output(
            self.l_bottleneck,
            {self.l_obs: high_obs_var, self.l_bottleneck_epsilon: state_info_vars["bottleneck_epsilon"]}
        )

    def get_all_probs(self, obs_var, state_info_vars):
        high_obs = obs_var[:, :self.observation_space.flat_dim - self.subgoal_dim]
        prob_vars = L.get_output(
            self.l_probs,
            {
                self.l_obs: tf.cast(high_obs, tf.float32),
                self.l_bottleneck_epsilon: state_info_vars["bottleneck_epsilon"]
            }
        )
        return prob_vars

    def get_subgoal_probs(self, all_probs, subgoals):
        return tf.batch_matmul(
            tf.expand_dims(subgoals, 1),
            tf.transpose(tf.pack(all_probs), (1, 0, 2))
        )[:, 0, :]

    def dist_info_sym(self, obs_var, state_info_vars):
        high_obs = obs_var[:, :self.observation_space.flat_dim - self.subgoal_dim]
        subgoals = obs_var[:, self.observation_space.flat_dim - self.subgoal_dim:]
        prob_vars = self.get_all_probs(high_obs, state_info_vars)
        probs = self.get_subgoal_probs(prob_vars, subgoals)
        return dict(prob=probs)

    def dist_info(self, obs, state_infos):
        return dict(prob=self.f_prob(obs, state_infos["bottleneck_epsilon"]))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    def get_action(self, observation):
        bottleneck_epsilon = np.random.normal(size=(self.bottleneck_dim,))
        flat_obs = self.observation_space.flatten(observation)
        dist_info = self.dist_info([flat_obs], dict(bottleneck_epsilon=[bottleneck_epsilon]))
        act = special.weighted_sample(dist_info["prob"], range(self.action_space.n))
        return act, dist_info

    def get_actions(self, observations):
        N = len(observations)
        bottleneck_epsilon = np.random.normal(size=(N, self.bottleneck_dim))
        flat_obses = self.observation_space.flatten_n(observations)
        dist_info = self.dist_info(flat_obses, dict(bottleneck_epsilon=bottleneck_epsilon))
        act = [special.weighted_sample(p, range(self.action_space.n)) for p in dist_info["prob"]]
        return act, dist_info

    @property
    def distribution(self):
        return self.dist


class IgnorantBranchingCategoricalMLPPolicy(BranchingCategoricalMLPPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            # shared_network,
            subgoal_dim,
            bottleneck_dim,
            bottleneck_std_threshold=1e-3,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
    ):
        Serializable.quick_init(self, locals())
        l_in = L.InputLayer(shape=(None, env_spec.observation_space.components[0].flat_dim), name="input")
        slice_start = env_spec.observation_space.components[0].components[0].flat_dim
        slice_end = env_spec.observation_space.components[0].flat_dim

        l_sliced_in = L.SliceLayer(l_in, name="sliced_input", indices=slice(slice_start, slice_end), axis=-1)

        shared_network = MLP(
            input_shape=(slice_end - slice_start,),
            input_layer=l_sliced_in,
            hidden_sizes=(100, 100),
            hidden_nonlinearity=tf.nn.tanh,
            output_dim=slice_end - slice_start,
            output_nonlinearity=tf.nn.tanh,
            name="dummy_shared",
        )
        BranchingCategoricalMLPPolicy.__init__(self, name=name, env_spec=env_spec, subgoal_dim=subgoal_dim,
                                               bottleneck_dim=bottleneck_dim, shared_network=shared_network,
                                               bottleneck_std_threshold=bottleneck_std_threshold,
                                               hidden_sizes=hidden_sizes, hidden_nonlinearity=hidden_nonlinearity)


def merge_grads(grads, *extra_grads_list):
    grad_dict = OrderedDict([(y, x) for x, y in grads])
    for extra_grads in extra_grads_list:
        for grad, var in extra_grads:
            if var not in grad_dict:
                grad_dict[var] = grad
            else:
                if grad is None:
                    pass
                elif grad_dict[var] is None:
                    grad_dict[var] = grad
                else:
                    grad_dict[var] += grad
    return [(y, x) for x, y in grad_dict.iteritems()]


UP = GridWorldEnv.action_from_direction("up")
DOWN = GridWorldEnv.action_from_direction("down")
LEFT = GridWorldEnv.action_from_direction("left")
RIGHT = GridWorldEnv.action_from_direction("right")


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
    env = TfEnv(CompoundActionSequenceEnv(wrapped_env, action_map, obs_include_history=True))
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
    return env, dict(
        observations=np.asarray(env.observation_space.flatten_n(observations)),
        actions=np.asarray(env.action_space.flatten_n(actions))
    )


class SeqGridExpert(object):
    def __init__(self, grid_size=5, action_map=None):
        assert grid_size == 5
        self.grid_size = grid_size
        if action_map is None:
            action_map = [
                [0, 1, 1],  # Left
                [1, 3, 3],  # Down
                [2, 2, 0],  # Right
                [3, 0, 2],  # Up
            ]
        self.action_map = action_map
        self.paths = None
        self.path_infos = None
        self.dataset = None
        self.envs = None
        self.seg_obs = None
        self.seg_actions = None

        base_map = [
            "SWGWW",
            "...WW",
            "WWWWW",
            "WWWWW",
            "WWWWW",
        ]
        # base_map = [["."] * grid_size for _ in range(grid_size)]
        # base_map[0][0] = 'S'
        # base_map[-1][-1] = 'G'
        wrapped_env = ImageGridWorld(desc=base_map)
        env = TfEnv(CompoundActionSequenceEnv(wrapped_env, action_map, obs_include_history=True))
        self.template_env = env

        self.env_spec = env.spec

    def build_dataset(self, batch_size):
        if self.dataset is None:
            paths = self.generate_demonstrations()
            observations = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
            actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
            N = observations.shape[0]
            seg_obs = observations.reshape((N / 3, 3, -1))
            seg_actions = actions.reshape((N / 3, 3, -1))
            self.dataset = BatchDataset([seg_obs, seg_actions], batch_size=batch_size)
            self.seg_obs = seg_obs
            self.seg_actions = seg_actions
        return self.dataset

    def generate_demonstrations(self):
        if self.paths is None:
            logger.log("generating paths")
            all_pos = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
            paths = []
            envs = []
            for start_pos, end_pos in itertools.permutations(all_pos, 2):
                # generate a demonstration trajectory from the start to the end
                env, path = generate_demonstration_trajectory(self.grid_size, start_pos, end_pos, self.action_map)
                paths.append(path)
                envs.append(env)
            logger.log("generated")
            # import ipdb; ipdb.set_trace()
            self.paths = paths
            self.envs = envs
        return self.paths

    def log_diagnostics(self, algo):
        logger.log("logging MI...")
        self.log_mis(algo)
        # logger.log("logging train stats...")
        # self.log_train_stats(algo)
        logger.log("logging test stats...")
        self.log_test_stats(algo)
        logger.log("logging I(g;s'|s)...")
        self.log_mi_goal_state(algo)
        logger.log("logging exact_H(g|z)...")
        self.log_exact_ent_g_given_z(algo)

    def log_mis(self, algo):

        low_policy = algo.low_policy

        # flat_obs =
        obs_dim = self.env_spec.observation_space.flat_dim

        first_obs = self.seg_obs[:, 0, :]
        all_obs = self.seg_obs.reshape((-1, obs_dim))
        # actions = self.seg_actions[:, 0, :]
        N = all_obs.shape[0]

        all_low_probs = []

        bottleneck_epsilon = np.random.normal(size=(N, algo.bottleneck_dim))
        # use shared random numbers to reduce variance
        for g in xrange(algo.subgoal_dim):
            subgoals = np.tile(
                np.asarray(algo.high_policy.action_space.flatten(g)).reshape((1, -1)),
                (N, 1)
            )
            low_obs = np.concatenate([all_obs, subgoals], axis=-1)
            low_probs = algo.low_policy.dist_info(low_obs, dict(bottleneck_epsilon=bottleneck_epsilon))["prob"]
            all_low_probs.append(low_probs)

        all_low_probs = np.asarray(all_low_probs)
        subgoals, _ = algo.high_policy.get_actions(self.env_spec.observation_space.unflatten_n(first_obs))
        subgoals = np.tile(
            np.asarray(subgoals).reshape((-1, 1)),
            (1, algo.subgoal_interval)
        ).flatten()
        subgoal_low_probs = all_low_probs[subgoals, np.arange(N)]

        # flat_low_probs = all_low_probs.reshape((-1, algo.low_policy.action_space.n))

        bottlenecks = algo.low_policy.f_bottleneck(all_obs, bottleneck_epsilon)

        p_g_given_z = algo.g_given_z_regressor._f_prob(bottlenecks)
        p_a_given_g_z = np.asarray(all_low_probs)

        p_a_given_z = np.sum((p_g_given_z.T[:, :, np.newaxis] * p_a_given_g_z), axis=0)

        ents_a_given_g_z = [
            algo.low_policy.distribution.entropy(dict(prob=cond_low_probs))
            for cond_low_probs in all_low_probs
            ]
        # import ipdb;
        # ipdb.set_trace()

        # p_a_given_s = np.mean(all_low_probs, axis=0)
        ent_a_given_z = np.mean(algo.low_policy.distribution.entropy(dict(prob=p_a_given_z)))
        ent_a_given_g_z = np.mean(np.sum(p_g_given_z.T * np.asarray(ents_a_given_g_z), axis=0))
        ent_a_given_subgoal_z = np.mean(algo.low_policy.distribution.entropy(dict(prob=subgoal_low_probs)))

        mi_a_g_given_z = ent_a_given_z - ent_a_given_g_z

        logger.record_tabular("exact_I(a;g|z)", mi_a_g_given_z)
        logger.record_tabular("exact_H(a|z)", ent_a_given_z)
        logger.record_tabular("exact_H(a|g,z)", ent_a_given_g_z)
        logger.record_tabular("exact_H(a|taken_g,z)", ent_a_given_subgoal_z)

    #
    # def log_train_stats(self, algo):
    #     env_spec = self.env_spec
    #     trained_policy = FixedClockPolicy(env_spec=env_spec, high_policy=algo.high_policy, low_policy=algo.low_policy,
    #                                       subgoal_interval=algo.subgoal_interval)
    #
    #     n_envs = len(self.envs)
    #
    #     train_venv = DummyVecEnv(env=self.template_env, n=n_envs, envs=self.envs,
    #                              max_path_length=algo.max_path_length)
    #
    #     path_rewards = [None] * n_envs
    #     path_discount_rewards = [None] * n_envs
    #     obses = train_venv.reset()
    #     dones = np.asarray([True] * n_envs)
    #     for t in xrange(algo.max_path_length):
    #         trained_policy.reset(dones)
    #         acts, _ = trained_policy.get_actions(obses)
    #         next_obses, rewards, dones, _ = train_venv.step(acts)
    #         obses = next_obses
    #         for idx, done in enumerate(dones):
    #             if done and path_rewards[idx] is None:
    #                 path_rewards[idx] = rewards[idx]
    #                 path_discount_rewards[idx] = rewards[idx] * (algo.discount ** t)
    #
    #     logger.record_tabular("AverageTrainReturn", np.mean(path_rewards))
    #     logger.record_tabular("AverageTrainDiscountedReturn", np.mean(path_discount_rewards))
    #
    def log_mi_goal_state(self, algo):
        # Essentially, we want to check how well the low-level policy learns the subgoals
        if not hasattr(self, "all_flat_obs"):
            all_obs = []
            all_desired_actions = []
            for e in self.envs:
                for nav_action, action_seq in enumerate(self.action_map):
                    obs = e.reset()
                    for raw_action in action_seq:
                        next_obs, _, _, _ = e.step(raw_action)
                        all_obs.append(obs)
                        all_desired_actions.append(raw_action)
                        obs = next_obs
            self.all_flat_obs = self.env_spec.observation_space.flatten_n(all_obs)
            self.all_desired_actions = np.asarray(all_desired_actions)

        all_flat_obs = self.all_flat_obs
        all_desired_actions = self.all_desired_actions
        N = all_flat_obs.shape[0]
        all_bottleneck_epsilons = np.random.normal(size=(N, algo.bottleneck_dim))

        subgoal_all_nav_action_probs = []
        subgoal_ents = []

        for subgoal in xrange(algo.subgoal_dim):
            subgoal_onehot = np.eye(algo.subgoal_dim, dtype=np.float32)[subgoal]
            all_low_obs = np.concatenate(
                [all_flat_obs, np.tile(subgoal_onehot.reshape((1, -1)), (N, 1))],
                axis=-1
            )
            action_probs = algo.low_policy.dist_info(all_low_obs, dict(bottleneck_epsilon=all_bottleneck_epsilons))[
                'prob']
            nav_action_probs = np.prod(action_probs[np.arange(N), all_desired_actions].reshape((-1, 4, 3)), axis=-1)
            dummy_action_prob = np.maximum(1e-8, 1. - np.sum(nav_action_probs, axis=-1))
            all_nav_action_probs = np.concatenate([nav_action_probs, dummy_action_prob.reshape((-1, 1))], axis=-1)
            subgoal_all_nav_action_probs.append(all_nav_action_probs)

            subgoal_ents.append(np.mean(Categorical(5).entropy(dict(prob=all_nav_action_probs))))

        marginal_all_nav_action_probs = np.mean(subgoal_all_nav_action_probs, axis=0)
        marginal_ent = np.mean(Categorical(5).entropy(dict(prob=marginal_all_nav_action_probs)))
        mi = marginal_ent - np.mean(subgoal_ents)

        logger.record_tabular("exact_I(g;s'|s)", mi)

    def log_test_stats(self, algo):
        env_spec = self.env_spec
        test_policy = FixedClockPolicy(env_spec=env_spec, high_policy=algo.alt_high_policy, low_policy=algo.low_policy,
                                       subgoal_interval=algo.subgoal_interval)

        n_envs = 100

        test_venv = DummyVecEnv(env=self.template_env, n=n_envs, max_path_length=algo.max_path_length)

        path_rewards = [None] * n_envs
        path_discount_rewards = [None] * n_envs
        obses = test_venv.reset()
        dones = np.asarray([True] * n_envs)
        for t in xrange(algo.max_path_length):
            test_policy.reset(dones)
            acts, _ = test_policy.get_actions(obses)
            next_obses, rewards, dones, _ = test_venv.step(acts)
            obses = next_obses
            for idx, done in enumerate(dones):
                if done and path_rewards[idx] is None:
                    path_rewards[idx] = rewards[idx]
                    path_discount_rewards[idx] = rewards[idx] * (algo.discount ** t)

        logger.record_tabular("AverageTestReturn", np.mean(path_rewards))
        logger.record_tabular("AverageTestDiscountedReturn", np.mean(path_discount_rewards))

    def log_exact_ent_g_given_z(self, algo):

        obs_dim = self.seg_obs.shape[-1]
        obs = self.seg_obs[:, 0, :]
        unflat_obs = self.env_spec.observation_space.unflatten_n(obs)

        all_obs = self.seg_obs.reshape((-1, obs_dim))

        n_samples = 1

        all_bottlenecks = []
        all_subgoals = []
        for _ in range(n_samples):
            subgoals = np.asarray(algo.high_policy.get_actions(unflat_obs)[0])
            flat_subgoals = algo.high_policy.action_space.flatten_n(subgoals)
            bottleneck_epsilon = np.random.normal(size=(all_obs.shape[0], algo.bottleneck_dim))
            bottlenecks = algo.low_policy.f_bottleneck(all_obs, bottleneck_epsilon)
            all_bottlenecks.extend(bottlenecks)
            all_subgoals.extend(np.tile(
                flat_subgoals.reshape((-1, algo.subgoal_dim, 1)),
                (1, 1, algo.subgoal_interval)
            ).reshape((-1, algo.subgoal_dim)))

        algo.g_given_z_regressor.fit(np.asarray(all_bottlenecks), np.asarray(all_subgoals))
        ent = np.mean(
            algo.g_given_z_regressor._dist.entropy(
                dict(prob=algo.g_given_z_regressor._f_prob(all_bottlenecks))
            )
        )
        logger.record_tabular("exact_H(g|z)", ent)
        bottleneck_dist_info = algo.low_policy.f_bottleneck_dist_info(all_obs)
        logger.record_tabular("exact_H(z|s)", np.mean(algo.low_policy.bottleneck_dist.entropy(bottleneck_dist_info)))
        logger.record_tabular("AverageBottleneckStd", np.mean(np.exp(bottleneck_dist_info["log_std"])))
        logger.record_tabular("MaxBottleneckStd", np.max(np.exp(bottleneck_dist_info["log_std"])))
        logger.record_tabular("MinBottleneckStd", np.min(np.exp(bottleneck_dist_info["log_std"])))


class FixedClockImitation(RLAlgorithm):
    def __init__(
            self,
            policy_module=None,
            policy_module_cls=None,
            subgoal_dim=4,
            subgoal_interval=3,
            bottleneck_dim=10,
            batch_size=500,
            learning_rate=1e-3,
            discount=0.99,
            max_path_length=100,
            n_epochs=100,
            mi_coeff=0.,
    ):
        self.env_expert = SeqGridExpert()
        if policy_module is None:
            if policy_module_cls is None:
                policy_module_cls = SeqGridPolicyModule
            policy_module = policy_module_cls()
        self.policy_module = policy_module

        self.subgoal_dim = subgoal_dim
        self.subgoal_interval = subgoal_interval
        self.bottleneck_dim = bottleneck_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.max_path_length = max_path_length
        self.n_epochs = n_epochs
        self.mi_coeff = mi_coeff

        env_spec = self.env_expert.env_spec
        self.high_policy = self.policy_module.new_high_policy(
            env_spec=env_spec,
            subgoal_dim=self.subgoal_dim
        )
        self.alt_high_policy = self.policy_module.new_alt_high_policy(
            env_spec=env_spec,
            subgoal_dim=self.subgoal_dim
        )
        self.low_policy = self.policy_module.new_low_policy(
            env_spec=env_spec,
            subgoal_dim=self.subgoal_dim,
            bottleneck_dim=self.bottleneck_dim
        )

        self.g_given_z_regressor = CategoricalMLPRegressor(
            name="g_given_z_regressor",
            input_shape=(bottleneck_dim,),
            output_dim=subgoal_dim,
            use_trust_region=False,
            hidden_sizes=(200, 200),
            optimizer=FirstOrderOptimizer(max_epochs=10, verbose=True),
        )

        self.logging_info = []

        self.f_train = None

    def surr_vlb_sym(self, obs_var, action_var, bottleneck_epsilon_var):
        """
        Compute the variational lower bound of p(action|state)
        """
        env_spec = self.env_expert.env_spec
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        flat_obs_var = tf.reshape(obs_var, (-1, obs_dim))
        flat_action_var = tf.reshape(action_var, (-1, action_dim))

        bottleneck_epsilon_var = tf.reshape(bottleneck_epsilon_var, (-1, self.bottleneck_dim))

        high_obs_var = obs_var[:, 0, :]
        policy_subgoal_dist = self.high_policy.dist_info_sym(high_obs_var, dict())

        flat_N = tf.shape(flat_obs_var)[0]

        all_sum_action_logli = []

        for subgoal in xrange(self.subgoal_dim):
            flat_subgoal = tf.tile(
                np.eye(self.subgoal_dim, dtype=np.float32)[subgoal].reshape((1, -1)),
                tf.pack([flat_N, 1]),
            )
            low_obs = tf.concat(1, [flat_obs_var, flat_subgoal])
            action_dist_info = self.low_policy.dist_info_sym(low_obs, dict(bottleneck_epsilon=bottleneck_epsilon_var))
            flat_action_logli = self.low_policy.distribution.log_likelihood_sym(flat_action_var, action_dist_info)
            action_logli = tf.reshape(flat_action_logli, (-1, self.subgoal_interval))
            sum_action_logli = tf.reduce_sum(action_logli, -1)
            all_sum_action_logli.append(sum_action_logli)

        vlb = tf.reduce_mean(tf.log(tf.reduce_sum(policy_subgoal_dist['prob'] * tf.exp(tf.transpose(tf.pack(
            all_sum_action_logli))), reduction_indices=-1)))

        # subgoal_kl = self.high_policy.distribution.kl_sym(
        #     recog_subgoal_dist,
        #     policy_subgoal_dist
        # )

        # E_sum_action_logli = tf.reduce_sum(recog_subgoal_dist['prob'] * tf.transpose(tf.pack(
        #     all_sum_action_logli)), reduction_indices=-1)
        #
        # vlb = tf.reduce_mean(E_sum_action_logli) - tf.reduce_mean(subgoal_kl)  # + \
        # 0.1 * tf.reduce_mean(self.recog.distribution.entropy_sym(recog_subgoal_dist))

        surr_vlb = vlb

        self.logging_info.extend([
            # ("average_H(q(g|s,a))", tf.reduce_mean(self.recog.distribution.entropy_sym(recog_subgoal_dist))),
            ("average_H(p(g|s))", tf.reduce_mean(self.high_policy.distribution.entropy_sym(policy_subgoal_dist))),
            # ("average_H(p(g|s))", tf.reduce_mean(self.high_policy.distribution.entropy_sym(policy_subgoal_dist))),
            # ("average_KL(q(g|s,a)||p(g|s))", tf.reduce_mean(subgoal_kl)),
            # ("average_E[log(p(a|g,s))]", tf.reduce_mean(E_sum_action_logli)),
        ])

        return vlb, surr_vlb

    def mi_a_g_given_z_sym(self, obs_var, action_var, bottleneck_epsilon_var):
        env_spec = self.env_expert.env_spec
        obs_dim = env_spec.observation_space.flat_dim
        flat_obs_var = tf.reshape(obs_var, (-1, obs_dim))
        bottleneck_epsilon_var = tf.reshape(bottleneck_epsilon_var, (-1, self.bottleneck_dim))

        bottleneck_var = self.low_policy.bottleneck_sym(flat_obs_var, dict(bottleneck_epsilon=bottleneck_epsilon_var))
        recog_subgoal_dist = self.g_given_z_regressor.dist_info_sym(bottleneck_var)

        all_action_probs = self.low_policy.get_all_probs(flat_obs_var, dict(bottleneck_epsilon=bottleneck_epsilon_var))

        marginal_action_probs = tf.reduce_sum(
            tf.expand_dims(tf.transpose(recog_subgoal_dist["prob"]), -1) * tf.pack(all_action_probs),
            0
        )
        marginal_ent = self.high_policy.distribution.entropy_sym(dict(prob=marginal_action_probs))

        conditional_ents = [
            self.high_policy.distribution.entropy_sym(dict(prob=cond_action_probs))
            for cond_action_probs in all_action_probs
            ]

        mean_conditional_ents = tf.reduce_sum(tf.transpose(recog_subgoal_dist["prob"]) * tf.pack(conditional_ents), 0)

        mi_a_g_given_z = tf.reduce_mean(marginal_ent) - tf.reduce_mean(mean_conditional_ents)
        return mi_a_g_given_z

    def mean_kl_sym(self, obs_var, action_var, bottleneck_epsilon_var, high_old_dist_info_vars,
                    low_old_dist_info_vars_dict_list):
        env_spec = self.env_expert.env_spec
        obs_dim = env_spec.observation_space.flat_dim

        flat_obs_var = tf.reshape(obs_var, (-1, obs_dim))
        bottleneck_epsilon_var = tf.reshape(bottleneck_epsilon_var, (-1, self.bottleneck_dim))

        subgoal_dist = self.high_policy.dist_info_sym(obs_var[:, 0, :])

        high_kl = self.high_policy.distribution.kl_sym(high_old_dist_info_vars, subgoal_dist)

        flat_N = tf.shape(flat_obs_var)[0]

        low_kls = []

        for subgoal, low_old_dist_info_vars in zip(xrange(self.subgoal_dim), low_old_dist_info_vars_dict_list):
            flat_subgoal = tf.tile(
                np.eye(self.subgoal_dim, dtype=np.float32)[subgoal].reshape((1, -1)),
                tf.pack([flat_N, 1]),
            )
            low_obs = tf.concat(1, [flat_obs_var, flat_subgoal])
            action_dist_info = self.low_policy.dist_info_sym(low_obs, dict(bottleneck_epsilon=bottleneck_epsilon_var))
            low_kl = self.low_policy.distribution.kl_sym(low_old_dist_info_vars, action_dist_info)
            low_kls.append(low_kl)

        low_kls_var = tf.pack(low_kls)
        sum_low_kls = tf.reduce_sum(tf.reshape(low_kls_var, (self.subgoal_dim, -1, self.subgoal_interval)), -1)

        mean_low_kl = tf.reduce_mean(tf.reduce_sum(subgoal_dist["prob"] * tf.transpose(sum_low_kls), -1))

        mean_kl = tf.reduce_mean(high_kl) + mean_low_kl

        return mean_kl

    def init_opt(self):
        logger.log("setting up training")

        env_spec = self.env_expert.env_spec

        obs_var = env_spec.observation_space.new_tensor_variable(
            name="obs",
            extra_dims=2,
        )
        action_var = env_spec.action_space.new_tensor_variable(
            name="action",
            extra_dims=2,
        )
        bottleneck_epsilon_var = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.subgoal_interval, self.bottleneck_dim),
            name="bottleneck"
        )
        high_old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='high_old_%s' % k)
            for k, shape in self.high_policy.distribution.dist_info_specs
            }
        low_old_dist_info_vars_dict_list = [
            {
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='subgoal_%d_low_old_%s' % (subgoal, k))
                for k, shape in self.low_policy.distribution.dist_info_specs
                }
            for subgoal in xrange(self.subgoal_dim)
            ]
        high_old_dist_info_vars_list = [high_old_dist_info_vars[k] for k in
                                        self.high_policy.distribution.dist_info_keys]
        low_old_dist_info_vars_list = [
            low_old_dist_info_vars[k] for k in self.low_policy.distribution.dist_info_keys
            for low_old_dist_info_vars in low_old_dist_info_vars_dict_list
            ]

        vlb, surr_vlb = self.surr_vlb_sym(obs_var, action_var, bottleneck_epsilon_var)
        mi_a_g_given_z = self.mi_a_g_given_z_sym(obs_var, action_var, bottleneck_epsilon_var)

        target = JointParameterized([self.high_policy, self.low_policy])
        loss = -surr_vlb - self.mi_coeff * mi_a_g_given_z
        # all_params = JointParameterized([self.high_policy, self.low_policy]).get_params(trainable=True)
        # bottleneck_params = L.get_all_params(self.low_policy.l_bottleneck, trainable=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # vlb_grads = optimizer.compute_gradients(-surr_vlb, var_list=all_params)
        # bottleneck_grads = optimizer.compute_gradients(
        #     - self.mi_coeff * mi_a_g_given_z, var_list=all_params  # bottleneck_params
        # )
        #
        # all_grads = merge_grads(vlb_grads, bottleneck_grads)

        from sandbox.rocky.straw.optimizers.tf_conjugate_gradient_optimizer import ConjugateGradientOptimizer, \
            FiniteDifferenceHvp

        mean_kl = self.mean_kl_sym(obs_var, action_var, bottleneck_epsilon_var, high_old_dist_info_vars,
                                   low_old_dist_info_vars_dict_list)

        optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        optimizer.update_opt(loss, target, (mean_kl, self.learning_rate),
                             [obs_var, action_var,
                              bottleneck_epsilon_var] + high_old_dist_info_vars_list + low_old_dist_info_vars_list)

        self.optimizer = optimizer

        # train_op = optimizer.apply_gradients(all_grads)

        self.logging_info.extend([
            ("average_NegVlb", -vlb),
            ("average_Vlb", vlb),
            ("average_I(a;g|z)", mi_a_g_given_z),
        ])
        #
        # self.f_train = tensor_utils.compile_function(
        #     inputs=[obs_var, action_var, bottleneck_epsilon_var],
        #     outputs=[train_op] + [x[1] for x in self.logging_info],
        # )

        self.f_train_log = tensor_utils.compile_function(
            inputs=[obs_var, action_var, bottleneck_epsilon_var],
            outputs=[x[1] for x in self.logging_info],
        )

    def get_snapshot(self):
        return dict(
            env=self.env_expert.template_env,
            policy=FixedClockPolicy(self.env_expert.env_spec, self.high_policy, self.low_policy, self.subgoal_interval),
            high_policy=self.high_policy,
            low_policy=self.low_policy,
        )

    def train(self):
        dataset = self.env_expert.build_dataset(self.batch_size)
        self.init_opt()
        with tf.Session() as sess:
            logger.log("initializing variables")
            sess.run(tf.initialize_all_variables())
            logger.log("initialized")

            obs_dim = self.env_expert.seg_obs.shape[-1]

            for epoch_id in xrange(self.n_epochs):

                logger.log("Start epoch %d..." % epoch_id)
                logger.record_tabular("Epoch", epoch_id)

                batch_obs = self.env_expert.seg_obs
                batch_actions = self.env_expert.seg_actions

                N = batch_obs.shape[0]
                batch_epsilons = np.random.normal(size=(N, self.subgoal_interval, self.bottleneck_dim))

                high_obs = batch_obs[:, 0, :]
                high_dist = self.high_policy.dist_info(high_obs)

                all_obs = np.reshape(batch_obs, (-1, obs_dim))

                N = all_obs.shape[0]

                all_low_dist_vars = []

                for g in xrange(self.subgoal_dim):
                    subgoals = np.tile(
                        np.asarray(self.high_policy.action_space.flatten(g)).reshape((1, -1)),
                        (N, 1)
                    )
                    low_obs = np.concatenate([all_obs, subgoals], axis=-1)
                    low_dist = self.low_policy.dist_info(low_obs, dict(bottleneck_epsilon=batch_epsilons.reshape((-1,
                                                                                               self.bottleneck_dim))))
                    all_low_dist_vars.extend([low_dist[k] for k in self.low_policy.distribution.dist_info_keys])

                high_dist_vars = [high_dist[k] for k in self.high_policy.distribution.dist_info_keys]

                all_inputs = [batch_obs, batch_actions, batch_epsilons] + high_dist_vars + all_low_dist_vars
                logger.record_tabular("MeanKLBefore", self.optimizer.constraint_val(all_inputs))
                self.optimizer.optimize(all_inputs)
                logger.record_tabular("MeanKLAfter", self.optimizer.constraint_val(all_inputs))

                logger.log("Evaluating...")

                all_vals = [self.f_train_log(batch_obs, batch_actions, batch_epsilons)]
                mean_all_vals = np.mean(np.asarray(all_vals), axis=0)
                for (k, _), v in zip(self.logging_info, mean_all_vals):
                    logger.record_tabular(k, v)
                self.env_expert.log_diagnostics(self)
                logger.dump_tabular()
                logger.save_itr_params(epoch_id, self.get_snapshot())
