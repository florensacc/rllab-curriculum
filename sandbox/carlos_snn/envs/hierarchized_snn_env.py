import numpy as np
from sandbox.carlos_snn.envs.mujoco.maze.maze_env import MazeEnv

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from rllab.misc import tensor_utils

from sandbox.carlos_snn.sampler.utils import rollout  # this is a different rollout! (not doing the same: no reset!)
from sandbox.carlos_snn.old_my_snn.hier_snn_mlp_policy import GaussianMLPPolicy_snn_hier

import joblib
import json
from rllab import config
import os


class HierarchizedSnnEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            time_steps_agg=1,
            pkl_path=None,
            json_path=None,
            npz_path=None,
            animate=False,
    ):
        ProxyEnv.__init__(self, env)
        Serializable.quick_init(self, locals())
        self.time_steps_agg = time_steps_agg
        self.animate = animate
        if json_path:
            self.data = json.load(open(os.path.join(config.PROJECT_PATH, json_path), 'r'))
            self.low_policy_latent_dim = self.data['json_args']['policy']['latent_dim']
        elif pkl_path:
            pkl_path = os.path.join(config.PROJECT_PATH, pkl_path)
            self.data = joblib.load(pkl_path)
            self.low_policy_latent_dim = self.data['policy'].latent_dim
        else:
            raise Exception("No path to file given")

        # assert isinstance(env, MazeEnv) or isinstance(env.wrapped_env,
        #                                               MazeEnv), "the obsSpaces mismatch but it's not a maze (by Carlos)"
        # I need to define a new hier-policy that will cope with that!
        self.low_policy = GaussianMLPPolicy_snn_hier(
            env_spec=env.spec,
            env=env,
            pkl_path=pkl_path,
            json_path=json_path,
            npz_path=npz_path,
            trainable_snn=False,
            external_latent=True,
        )

    @property
    @overrides
    def action_space(self):
        lat_dim = self.low_policy_latent_dim
        return spaces.Discrete(lat_dim)  # the action is now just a selection

    @overrides
    def step(self, action):
        action = self.action_space.flatten(action)
        with self.low_policy.fix_latent(action):
            # print("From hier_snn_env --> the hier action is prefixed latent: {}".format(self.low_policy.pre_fix_latent))
            frac_path = rollout(self.wrapped_env, self.low_policy, max_path_length=self.time_steps_agg,
                                animated=self.animate, speedup=1000)
            next_obs = frac_path['observations'][-1]
            reward = np.sum(frac_path['rewards'])
            done = self.time_steps_agg > len(
                frac_path['observations'])  # if the rollout was not maximal it was "done"!`
            # it would be better to add an extra flagg to this rollout to check if it was done in the last step
            last_agent_info = dict((k, val[-1]) for k, val in frac_path['agent_infos'].items())
            last_env_info = dict((k, val[-1]) for k, val in frac_path['env_infos'].items())
        # print("finished step of {}, with cummulated reward of: {}".format(len(frac_path['observations']), reward))
        # print("Next obs (com): {}, rew: {}, last_env_info: {}, last_agent_info: {}".format(last_env_info, reward, last_env_info,
        #                                                                              last_agent_info))
        if done:
            # print("\n ########## \n ***** done!! *****")
            # if done I need to PAD the tensor so there is no mismatch! Pad with what? with the last elem!
            full_path = tensor_utils.pad_tensor_dict(frac_path, self.time_steps_agg, mode='last')
        else:
            full_path = frac_path

        return Step(next_obs, reward, done,
                    last_env_info=last_env_info, last_agent_info=last_agent_info, full_path=full_path)
        # the last kwargs will all go to env_info, so path['env_info']['full_path'] gives a dict with the full path!

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        ## to use the visualization I need to append all paths!
        ## and also I need the paths to have the "agent_infos" key including the latent!!
        expanded_paths = [tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path']) for path in paths]
        self.wrapped_env.log_diagnostics(expanded_paths)

    def __str__(self):
        return "Hierarchized: %s" % self._wrapped_env


hierarchize_snn = HierarchizedSnnEnv
