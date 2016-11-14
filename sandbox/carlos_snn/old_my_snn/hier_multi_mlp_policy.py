import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import theano
import theano.tensor as TT
import numpy as np
from contextlib import contextmanager

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from sandbox.carlos_snn.core.lasagne_layers import CropLayer, ConstOutputLayer, SumProdLayer
from rllab.spaces import Box

from rllab.envs.normalized_env import NormalizedEnv  # this is just to check if the env passed is a normalized maze
from sandbox.carlos_snn.envs.mujoco.maze.maze_env import MazeEnv
from sandbox.carlos_snn.envs.mujoco.gather.gather_env import GatherEnv

from rllab.sampler.utils import \
    rollout  # I need this for logging the diagnostics: run the policy with all diff selectors

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.misc import autoargs
from rllab.distributions.diagonal_gaussian import DiagonalGaussian

import joblib
import json
import os
from rllab import config


class GaussianMLPPolicy_multi_hier(StochasticPolicy, LasagnePowered, Serializable):  # also inherits form Parametrized
    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('std_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected layers for std, note'
                       'there is a difference in semantics than above: here an empty'
                       'list means that std is independent of input and the last size is ignored')
    @autoargs.arg('initial_std', type=float,
                  help='Initial std')
    @autoargs.arg('std_trainable', type=bool,
                  help='Is std trainable')
    @autoargs.arg('output_nl', type=str,
                  help='nonlinearity for the output layer')
    @autoargs.arg('nonlinearity', type=str,
                  help='nonlinearity used for each hidden layer, can be one '
                       'of tanh, sigmoid')
    @autoargs.arg('bn', type=bool,
                  help='whether to apply batch normalization to hidden layers')
    def __init__(
            self,
            env_spec,
            env,
            pkl_paths=(),
            json_paths=(),
            npz_paths=(),
            trainable_old=True,
            hidden_sizes_old=(32, 32),
            hidden_sizes_selector=(10, 10),
            external_selector=False,
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            std_hidden_nonlinearity=NL.tanh,
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            min_std=1e-4,
    ):
        # define where are the old policies to use and what to do with them:
        self.trainable_old = trainable_old  # whether to keep training the old policies loaded here
        self.pkl_paths = pkl_paths
        self.json_paths = json_paths
        self.npz_paths = npz_paths
        self.selector_dim = max(len(json_paths), len(pkl_paths))  # pkl could be zero if giving npz
        # if not use a selector NN here, just externally fixed selector variable:
        self.external_selector = external_selector  # whether to use the selectorNN defined here or the pre_fix_selector
        self.pre_fix_selector = np.zeros((self.selector_dim))  # if this is not empty when using reset() it will use this selector
        self.selector_fix = np.zeros((self.selector_dim))  # this will hold the selectors variable sampled in reset()
        self.shared_selector_var = theano.shared(self.selector_fix)  # this is for external selector! update that
        # else, describe the MLP used:
        self.hidden_sizes_selector = hidden_sizes_selector  # size of the selector NN defined here
        self.min_std = min_std
        self._set_std_to_0 = False

        self.action_dim = env_spec.action_space.flat_dim  # not checking that all the old policies have this act_dim

        self.old_hidden_sizes = []
        # self.old_min_stds = []
        # self.old_adaptive_stds = []
        # self.old_std_hidden_sizes = []
        # assume json always given
        for json_path in self.json_paths:
            data = json.load(open(os.path.join(config.PROJECT_PATH, json_path), 'r'))
            old_json_policy = data['json_args']["policy"]
            self.old_hidden_sizes.append(old_json_policy['hidden_sizes'])
            # self.old_min_stds.append(old_json_policy['min_std'])
            # self.old_adaptive_stds.append(old_json_policy['adaptive_std'])
            # self.old_std_hidden_sizes.append(old_json_policy['std_hidden_sizes'])
        print("Final attributes: ", self.selector_dim, self.old_hidden_sizes)

        # retrieve dimensions and check consistency
        if isinstance(env, MazeEnv) or isinstance(env, GatherEnv):
            self.obs_robot_dim = env.robot_observation_space.flat_dim
            self.obs_maze_dim = env.maze_observation_space.flat_dim
        elif isinstance(env, NormalizedEnv):
            if isinstance(env.wrapped_env, MazeEnv) or isinstance(env.wrapped_env, GatherEnv):
                self.obs_robot_dim = env.wrapped_env.robot_observation_space.flat_dim
                self.obs_maze_dim = env.wrapped_env.maze_observation_space.flat_dim
            else:
                self.obs_robot_dim = env.wrapped_env.observation_space.flat_dim
                self.obs_maze_dim = 0
        else:
            self.obs_robot_dim = env.observation_space.flat_dim
            self.obs_maze_dim = 0
        print("the dims of the env are(rob/maze): ", self.obs_robot_dim, self.obs_maze_dim)
        all_obs_dim = env_spec.observation_space.flat_dim
        assert all_obs_dim == self.obs_robot_dim + self.obs_maze_dim
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        if self.external_selector:  # in case we want to fix the selector externally
            l_all_obs_var = L.InputLayer(shape=(None,) + (self.obs_robot_dim + self.obs_maze_dim,))
            all_obs_var = l_all_obs_var.input_var
            # l_selection = ConstOutputLayer(incoming=l_all_obs_var, output_var=self.shared_selector_var)
            l_selection = ParamLayer(incoming=l_all_obs_var, num_units=self.selector_dim, param=self.shared_selector_var,
                                     trainable=False)
            selection_var = L.get_output(l_selection)
        else:
            # create network with softmax output: it will be the selector!
            selector_network = MLP(
                input_shape=(self.obs_robot_dim + self.obs_maze_dim,),
                output_dim=self.selector_dim,
                hidden_sizes=self.hidden_sizes_selector,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=NL.softmax,
            )
            l_all_obs_var = selector_network.input_layer
            all_obs_var = selector_network.input_layer.input_var

            # collect the output to select the behavior of the robot controller (equivalent to selectors)
            l_selection = selector_network.output_layer
            selection_var = L.get_output(l_selection)

        # split all_obs into the robot and the maze obs --> ROBOT goes first!!
        l_obs_robot = CropLayer(l_all_obs_var, start_index=None, end_index=self.obs_robot_dim)
        l_obs_maze = CropLayer(l_all_obs_var, start_index=self.obs_robot_dim, end_index=None)

        obs_robot_var = all_obs_var[:, :self.obs_robot_dim]
        obs_maze_var = all_obs_var[:, self.obs_robot_dim:]

        # create the action networks
        self.old_l_means = []  # I do this self in case I wanna access it from reset
        self.old_l_log_stds = []
        self.old_layers = []
        for i in range(self.selector_dim):
            mean_network = MLP(
                input_layer=l_obs_robot,
                output_dim=self.action_dim,
                hidden_sizes=self.old_hidden_sizes[i],
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                name="meanMLP{}".format(i),
            )
            self.old_l_means.append(mean_network.output_layer)
            self.old_layers += mean_network.layers

            # if self.old_adaptive_stds[i]:
            #     log_std_network = MLP(
            #         input_layer=l_obs_robot,
            #         output_dim=self.action_dim,
            #         hidden_sizes=self.old_std_hidden_sizes[i],
            #         hidden_nonlinearity=std_hidden_nonlinearity,
            #         output_nonlinearity=None,
            #         name="log_stdMLP{}".format(i),
            #     )
            #     self.old_l_log_stds.append(log_std_network.output_layer)
            #     self.old_layers += log_std_network.layers
            # else:
            l_log_std = ParamLayer(
                incoming=mean_network.input_layer,
                num_units=self.action_dim,
                param=lasagne.init.Constant(np.log(init_std)),
                name="output_log_std{}".format(i),
                trainable=learn_std,
            )
            self.old_l_log_stds.append(l_log_std)
            self.old_layers += [l_log_std]

        if not self.trainable_old:
            for layer in self.old_layers:
                for param, tags in layer.params.items():  # params of layer are OrDict: key=the shared var, val=tags
                    tags.remove("trainable")

        if self.json_paths and self.npz_paths:
            old_params_dict = {}
            for i, npz_path in enumerate(self.npz_paths):
                params_dict = dict(np.load(os.path.join(config.PROJECT_PATH, npz_path)))
                renamed_warm_params_dict = {}
                for key in params_dict.keys():
                    if key == 'output_log_std.param':
                        old_params_dict['output_log_std{}.param'.format(i)] = params_dict[key]
                    elif 'meanMLP_' == key[:8]:
                        old_params_dict['meanMLP{}_'.format(i)+key[8:]] = params_dict[key]
                    else:
                        old_params_dict['meanMLP{}_'.format(i)+key] = params_dict[key]
            self.set_old_params(old_params_dict)

        elif self.pkl_paths:
            old_params_dict = {}
            for i, pkl_path in enumerate(self.pkl_paths):
                data = joblib.load(os.path.join(config.PROJECT_PATH, pkl_path))
                params = data['policy'].get_params_internal()
                for param in params:
                    if param.name == 'output_log_std.param':
                        old_params_dict['output_log_std{}.param'.format(i)] = param.get_value()
                    elif 'meanMLP_' == param.name[:8]:
                        old_params_dict['meanMLP{}_'.format(i)+param.name[8:]] = param.get_value()
                    else:
                        old_params_dict['meanMLP{}_'.format(i)+param.name] = param.get_value()
            self.set_old_params(old_params_dict)

        # new layers actually selecting the correct output
        l_mean = SumProdLayer(self.old_l_means + [l_selection])
        l_log_std = SumProdLayer(self.old_l_log_stds + [l_selection])
        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))

        self._l_mean = l_mean
        self._l_log_std = l_log_std

        self._dist = DiagonalGaussian(self.action_dim)

        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(GaussianMLPPolicy_multi_hier, self).__init__(env_spec)

        self._f_old_means = ext.compile_function(
            inputs=[all_obs_var],
            outputs=[L.get_output(l_old_mean) for l_old_mean in self.old_l_means]
        )

        self._f_all_inputs = ext.compile_function(
            inputs=[all_obs_var],
            outputs=[L.get_output(l_old_mean) for l_old_mean in self.old_l_means] + [selection_var]
        )

        self._f_dist = ext.compile_function(
            inputs=[all_obs_var],
            outputs=[mean_var, log_std_var],
        )
        # if I want to monitor the selector output
        self._f_select = ext.compile_function(
            inputs=[all_obs_var],
            outputs=selection_var,
        )

    def get_old_params(self):
        params = []
        for layer in self.old_layers:
            params += layer.get_params()
        return params

    # another way will be to do as in parametrized.py and flatten_tensors (in numpy). But with this I check names
    def set_old_params(self, old_params):
        if type(old_params) is dict:  # if the old_params are a dict with the param name as key and a numpy array as value
            params_value_by_name = old_params
        elif type(old_params) is list:  # if the old_params are a list of theano variables  **NOT CHECKING THIS!!**
            params_value_by_name = {}
            for param in old_params:
                params_value_by_name[param.name] = param.get_value()
        else:
            params_value_by_name = {}
            print("The old_params was not understood!")

        local_params = self.get_old_params()
        # print(local_params)
        for param in local_params:
            param.set_value(params_value_by_name[param.name])

    def dist_info_sym(self, obs_var, state_info_var=None):
        mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std], obs_var)
        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def get_actions(self, observations):
        selector_output = self._f_select(observations)
        mean, log_std = self._f_dist(observations)

        if self._set_std_to_0:
            actions = mean
            log_std = -1e6 * np.ones_like(log_std)
        else:
            rnd = np.random.normal(size=mean.shape)
            actions = rnd * np.exp(log_std) + mean
        # print(selector_output)
        return actions, dict(mean=mean, log_std=log_std, selectors=selector_output)

    def set_pre_fix_selector(self, selector):
        self.pre_fix_selector = np.array(selector)

    def unset_pre_fix_selector(self):
        self.pre_fix_selector = np.array([])

    @contextmanager
    def fix_selector(self, selector):
        self.pre_fix_selector = np.array(selector)
        yield
        self.pre_fix_selector = np.array([])

    @contextmanager
    def set_std_to_0(self):
        self._set_std_to_0 = True
        yield
        self._set_std_to_0 = False

    @overrides
    def reset(self):  # executed at the start of every rollout. Will fix the selector if needed.
        if self.pre_fix_selector.size > 0:
            self.selector_fix = self.pre_fix_selector
        # this is needed for the external selector!!
        self.shared_selector_var.set_value(np.array(self.selector_fix))

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('MaxPolicyStd', np.max(np.exp(log_stds)))
        logger.record_tabular('MinPolicyStd', np.min(np.exp(log_stds)))
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        """
        We set the distribution to the policy itself since we need some behavior different from a usual diagonal
        Gaussian distribution.
        """
        return self._dist

    def log_likelihood(self, actions, agent_infos, action_only=True):
        # First compute logli of the action. This assumes the selectors FIX to whatever was sampled, and hence we only
        # need to use the mean and log_std, but not any information about the selectors
        logli = self._dist.log_likelihood(actions, agent_infos)
        if not action_only:
            raise NotImplementedError
            #   if not action_only:
            #       for idx, selector_name in enumerate(self._selector_distributions):
            #           selector_var = dist_info["selector_%d" % idx]
            #           prefix = "selector_%d_" % idx
            #           selector_dist_info = {k[len(prefix):]: v for k, v in dist_info.iteritems() if k.startswith(
            #               prefix)}
            #           logli += selector_name.log_likelihood(selector_var, selector_dist_info)
        return logli
