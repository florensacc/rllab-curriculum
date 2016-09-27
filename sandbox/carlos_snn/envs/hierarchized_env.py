import numpy as np
from sandbox.carlos_snn.envs.mujoco.maze.maze_env import MazeEnv

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

from sandbox.carlos_snn.sampler.utils import rollout  # this is a different rollout! (now doing the same, could change)
from sandbox.carlos_snn.old_my_snn.hier_mlp_policy import GaussianMLPPolicy_hier

import joblib


class HierarchizedEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            time_steps_agg=1,
            pkl_path=None,
    ):
        ProxyEnv.__init__(self, env)
        Serializable.quick_init(self, locals())
        self.time_steps_agg = time_steps_agg
        if pkl_path:
            self.data = joblib.load(pkl_path)
        else:
            raise Exception("No pkl path given")
        self.low_policy = self.data['policy']

        assert self.low_policy._env_spec.action_space.flat_dim == env.spec.action_space.flat_dim, "the action" \
            "space of the hierarchized env and the pre-trained policy do not coincide: might be different robot!"
        if self.low_policy._env_spec.observation_space.flat_dim != env.spec.observation_space.flat_dim:
            print ("The ObsSpace of hierarchized env is {}, the pre-trained was {}".format(
                env.spec.observation_space.flat_dim,
                self.low_policy._env_spec.observation_space.flat_dim))
            assert isinstance(env, MazeEnv) or isinstance(env.wrapped_env, MazeEnv), "the obsSpaces mismatch but it's not a maze (by Carlos)!!"
            # I need to define a new hier-policy that will cope with that!
            self.low_policy = GaussianMLPPolicy_hier(
                env_spec=env.spec,
                env=env,
                pkl_path=pkl_path,
                trainable_snn=False,
                external_latent=True,
            )

    @property
    @overrides
    def action_space(self):
        # print ("the action space of the hierarchyzed env is: {}".format(self.low_policy.latent_dim))
        lat_dim = self.low_policy.latent_dim
        return spaces.Discrete(lat_dim)  # the action is now just a selection

    @overrides
    def step(self, action):
        action = self.action_space.flatten(action)
        with self.low_policy.fix_latent(action):
            print ("The hier action is prefixed latent: {}".format(self.low_policy.pre_fix_latent))
            frac_path = rollout(self.wrapped_env, self.low_policy, max_path_length=self.time_steps_agg, animated=True, speedup=1000)
            next_obs = frac_path['observations'][-1]
            reward = np.sum(frac_path['rewards'])
            done = self.time_steps_agg > len(frac_path['observations'])  # if the rollout was not maximal it was "done"!`
            # it would be better to add an extra flagg to this rollout to check if it was done in the last step
            agent_info = dict((k, val[-1]) for k, val in frac_path['agent_infos'].iteritems())
            env_info = dict((k, val[-1]) for k, val in frac_path['env_infos'].iteritems())
        print ("finished step of {}, with cummulated reward of: {}".format(len(frac_path['observations']), reward))
        if done:
            print ("\n ########## \n ***** done!! *****")
        return Step(next_obs, reward, done, last_env_info=env_info, last_agent_info=agent_info, full_path=frac_path)
        # the last kwargs will all go to env_info

    @overrides
    def log_diagnostics(self, paths):
        ## to use the visualization I need to append all paths!
        ## and also I need the paths to have the "agent_infos" key including the latent!!
        # all_obs = [np.concatenate(env_info['full_path']['observations'] for env_info in path['env_infos']) for path in paths]
        # obs_by_steps = [env_info['full_path']['observations'] for path in paths for env_info in path['env_infos']]
        # lats = [agent_info['latent'] for path in paths for agent_info in path['agent_infos']]
        # chopped_paths = [{'observations': obs} for obs in obs_by_steps]
        expanded_paths = []
        for path in paths:
            for i, obs in enumerate(path['env_infos']['full_path']['observations']):
                lat = path['env_infos']['full_path']['agent_infos']['latents'][i]
                chunk_path = {'observations': obs, 'agent_infos': {'latents': lat}}
                expanded_paths.append(chunk_path)
        print (expanded_paths)
        self.wrapped_env.log_diagnostics(expanded_paths)

    def __str__(self):
        return "Hierarchized: %s" % self._wrapped_env


hierarchize = HierarchizedEnv
