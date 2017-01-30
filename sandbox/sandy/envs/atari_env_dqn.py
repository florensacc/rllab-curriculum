"""Wrapper around AtariEnv, to make it work with A3C implementation.
"""
import copy

from sandbox.sandy.envs.atari_env import AtariEnv
from sandbox.sandy.adversarial.shared import get_base_env

class AtariEnvDQN(AtariEnv):
    def __init__(self, env_name, record_video=False, video_schedule=None, \
                 log_dir=None, record_log=True, force_reset=False, **kwargs):

        AtariEnv.__init__(self, env_name, record_video=record_video, \
                          video_schedule=video_schedule, log_dir=log_dir, \
                          record_log=record_log, force_reset=force_reset, **kwargs)

        self.base_env = get_base_env(self.env)
        self.legal_actions = self.base_env.ale.getMinimalActionSet()

    @property
    def number_of_actions(self):
        return len(self.legal_actions)

    @property
    def minimal_action_set(self):
        return self.legal_actions

    @property
    def ale_screen_dims(self):
        return self.base_env.ale.getScreenDims()

    @property
    def ale_ram_size(self):
        return self.base_env.ale.getRAMSize()

    @property
    def last_state(self):
        #return self.unscale_obs(self.observation[-1,:,:])
        return self.observation[-1,:,:]

    def set_seed(self, seed):
        #self.base_env.ale.setInt(b'random_seed', seed)
        self.base_env._seed(int(seed))

    def get_ram(self):
        return self.base_env.ale.getRAM()

    def get_screen_rgb(self):
        return self.base_env.ale.getScreenRGB()

    def get_lives(self):
        return self.base_env.ale.lives()
