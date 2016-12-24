"""Wrapper around OpenAI Gym Atari environments. Similar to rllab.envs.gym_env.
   Pre-processes raw Atari frames (210x160x3) into 84x84x4 grayscale images
   across the last four timesteps."""

import gym, gym.envs, gym.spaces
import numpy as np
from rllab.envs.base import Step
from rllab.envs.gym_env import GymEnv
from rllab.misc.overrides import overrides
from rllab.spaces.box import Box
from scipy.misc import imresize

RESIZE_W = 42  # 84
RESIZE_H = 42  # 84
N_FRAMES = 4
SCALE = 255.0

RGB2Y_COEFF = np.array([0.2126, 0.7152, 0.0722])  # Y = np.dot(rgb, RGB2Y_COEFF)

class AtariEnv(GymEnv):
    def __init__(self, env_name, record_video=True, video_schedule=None, \
                 log_dir=None, record_log=True):
        super().__init__(env_name, record_video=record_video, \
                         video_schedule=video_schedule, log_dir=log_dir, \
                         record_log=record_log)

        # Overwrite self._observation_space since preprocessing changes it
        # and Theano requires axes to be in the order (batch size, # channels,
        # # rows, # cols) instead of (batch size, # rows, # cols, # channels)
        self._observation_space = Box(0.,1.,(N_FRAMES,RESIZE_H,RESIZE_W))
        self.update_curr_obs(None)

        # adversary_fn - function handle for adversarial perturbation of observation
        self.adversary_fn = None

    def set_adversary_fn(self, fn_handle):
        self.adversary_fn = fn_handle

    def clear_curr_obs(self):
        self._curr_obs = np.zeros(self.observation_space.shape)

    def update_curr_obs(self, next_obs):
        if next_obs is None:
            self.clear_curr_obs()
        else:
            next_obs = self.preprocess_obs(next_obs)
            self._curr_obs = np.r_[self._curr_obs[1:,:,:], next_obs[np.newaxis,:]]
            if self.adversary_fn is not None:
                # Adversarially perturb input
                self._curr_obs = self.adversary_fn(self._curr_obs)

    def preprocess_obs(self, obs):
        # Preprocess Atari frames based on released DQN code from Nature paper:
        #     1) Convert RGB to grayscale (Y in YUV)
        #     2) Rescale image to 84 x 84 using 'bilinear' method
        obs = np.dot(obs, RGB2Y_COEFF)  # Convert RGB to Y
        obs = imresize(obs, self.observation_space.shape[1:], interp='bilinear')
        return obs / SCALE  # Scale values to be from 0 to 1

    @overrides
    def step(self, action):
        # next_obs should be Numpy array of shape (210,160,3)
        next_obs, reward, done, info = self.env.step(action)
        self.update_curr_obs(next_obs)
        return Step(self._curr_obs, reward, done, **info)

    @overrides
    def reset(self):
        obs = self.env.reset()
        self.clear_curr_obs()
        self.update_curr_obs(obs)
        return self._curr_obs
