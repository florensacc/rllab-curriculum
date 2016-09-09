import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

from sandbox.carlos_snn.sampler.utils import rollout  # this is a different rollout! (now doing the same, could change)
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

    @property
    @overrides
    def action_space(self):
        # print "the action space of the hierarchyzed env is: {}".format(self.low_policy.latent_dim)
        lat_dim = self.low_policy.latent_dim
        return spaces.Discrete(lat_dim)  # the action is now just a selection

    @overrides
    def step(self, action):
        action = self.action_space.flatten(action)
        print("taking a step in hierarchized env:")
        with self.low_policy.fix_latent(action):
            print("the prefixed latent is: {}".format(self.low_policy.pre_fix_latent))
            frac_path = rollout(self.wrapped_env, self.low_policy, max_path_length=self.time_steps_agg)
            next_obs = frac_path['observations'][-1]
            reward = np.sum(frac_path['rewards'])
            done = self.time_steps_agg < len(frac_path['observations'])

            agent_info = dict((k, val[-1]) for k, val in frac_path['agent_infos'].items())
            env_info = dict((k, val[-1]) for k, val in frac_path['env_infos'].items())
        print("finished step of {}, with cummulated reward of: {}".format(self.time_steps_agg, reward))
        return Step(next_obs, reward, done, agent_info=agent_info, env_info=env_info)

    def __str__(self):
        return "Hierarchized: %s" % self._wrapped_env


hierarchize = HierarchizedEnv
