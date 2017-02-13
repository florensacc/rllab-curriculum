"""Wrapper around AtariEnv, to make it work with A3C implementation.
"""
import copy

from sandbox.sandy.envs.atari_env import AtariEnv, get_base_env
from sandbox.sandy.async_rl.shareable.base import Shareable
from sandbox.sandy.async_rl.utils.picklable import Picklable

class AtariEnvA3C(AtariEnv, Shareable, Picklable):
    def __init__(self, env_name, record_video=False, video_schedule=None, \
                 log_dir=None, record_log=True, force_reset=False, **kwargs):
 
        self.init_params = locals()
        self.init_params.pop('self')

        AtariEnv.__init__(self, env_name, record_video=record_video, \
                          video_schedule=video_schedule, log_dir=log_dir, \
                          record_log=record_log, force_reset=force_reset, **kwargs)

        self.base_env = get_base_env(self.env)
        self.legal_actions = self.base_env.ale.getMinimalActionSet()

    def set_seed(self, seed):
        #self.base_env.ale.setInt(b'random_seed', seed)
        self.base_env._seed(int(seed))

    def prepare_sharing(self):
        self.share_params = dict()

    def process_copy(self):
        """
        Simply create an env with the same initialization.
        """
        init_params_copy = copy.deepcopy(self.init_params)
        if 'kwargs' in init_params_copy:
            kwargs = init_params_copy.pop('kwargs')
            new_env = AtariEnvA3C(**init_params_copy, **kwargs)
        else:
            new_env = AtariEnvA3C(**init_params_copy)
        return new_env

    def receive_action(self, action):
        self.step(action)
        return self._reward

    def update_params(self, global_vars, training_args):
        pass

    def initialize(self):
        self.reset()

    @property
    def extra_infos(self):
        return {}

    @property
    def state(self):
        #return list(self.unscale_obs(self.observation))
        return list(self.observation)

    @property
    def number_of_actions(self):
        return len(self.legal_actions)
