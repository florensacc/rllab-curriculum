"""Wrapper around OpenAI Gym Atari environments. Similar to rllab.envs.gym_env.
   Pre-processes raw Atari frames (210x160x3) into 84x84x4 grayscale images
   across the last four timesteps."""

import cv2
import gym, gym.envs, gym.spaces
import numpy as np
from scipy.misc import imresize

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.spaces.box import Box
from sandbox.sandy.envs.gym_env import GymEnv
from sandbox.sandy.misc.util import row_concat

DEFAULT_IMG_HEIGHT = 42
DEFAULT_IMG_WIDTH = 42
DEFAULT_N_FRAMES = 4
DEFAULT_FRAMESKIP = 4
DEFAULT_PERSISTENT = True
DEFAULT_SCALE_NEG1_1 = True
SCALE = 255.0

RGB2Y_COEFF = np.array([0.2126, 0.7152, 0.0722])  # Y = np.dot(rgb, RGB2Y_COEFF)

def get_base_env(obj):
    # Find level of obj that contains base environment, i.e., the env that links to ALE
    # (New version of Monitor in OpenAI gym adds an extra level of wrapping)
    while True:
        if not hasattr(obj, 'env'):
            return None
        if hasattr(obj.env, 'ale'):
            return obj.env
        else:
            obj = obj.env

class AtariEnv(GymEnv):
    def __init__(self, env_name, record_video=False, video_schedule=None, \
                 log_dir=None, record_log=True, force_reset=False, **kwargs):
        # persistent = True if the adversarial changes should stay in the history
        #                   of N_FRAMES
        # A persistent adversary attacks before the observation is stored in history
        # A non-persistent adversary attacks after observation is stored in history.
        # before it is fed through the policy neural net (this seems contrived)

        super().__init__(env_name, record_video=record_video, \
                         video_schedule=video_schedule, log_dir=log_dir, \
                         record_log=record_log, force_reset=force_reset, **kwargs)

        if 'seed' in kwargs:
            self.env.env.env._seed(kwargs['seed'])
        self.img_height = kwargs.get('img_height', DEFAULT_IMG_HEIGHT)
        self.img_width = kwargs.get("img_width", DEFAULT_IMG_WIDTH)
        self.n_frames = kwargs.get("n_frames", DEFAULT_N_FRAMES)
        self.persistent_adv = kwargs.get("persistent", DEFAULT_PERSISTENT)
        self.scale_neg1_1 = kwargs.get("scale_neg1_1", DEFAULT_SCALE_NEG1_1)

        frameskip = kwargs.get('frame_skip', DEFAULT_FRAMESKIP)
        self.env.env.env.frameskip = frameskip

        # adversary_fn - function handle for adversarial perturbation of observation
        self.adversary_fn = None

        # Overwrite self._observation_space since preprocessing changes it
        # and Theano requires axes to be in the order (batch size, # channels,
        # # rows, # cols) instead of (batch size, # rows, # cols, # channels)
        if self.scale_neg1_1:
            self._observation_space = Box(-1.,1.,(self.n_frames,self.img_height,self.img_width))
        else:
            self._observation_space = Box(0.,1.,(self.n_frames,self.img_height,self.img_width))

        self.update_last_frames(None)
        self._is_terminal = True  # Need to call self.reset() to set this to False
        self._reward = 0

    @property
    def reward(self):
        return self._reward

    @property
    def is_terminal(self):
        return self._is_terminal

    @property
    def observation(self):
        if self.persistent_adv:
            return self.last_adv_frames
        else:
            return 
            return row_concat(self.last_frames[:-1,:,:], self.last_adv_frames[-1,:,:][np.newaxis,:,:])

    def set_adversary_fn(self, fn_handle):
        self.adversary_fn = fn_handle

    def clear_last_frames(self):
        # last_frames: last N_FRAMES of resized, scaled observations
        self.last_frames = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.last_frames = self.scale_obs(self.last_frames)

        # last_adv_frames: last N_FRAMES of resized, scaled, *and* 
        #                  adversarially perturbed observations
        self.last_adv_frames = np.array(self.last_frames)

    def update_last_frames(self, next_obs):
        if next_obs is None:
            self.clear_last_frames()
        else:
            next_obs = self.preprocess_obs(next_obs)
            self.last_frames = row_concat(self.last_frames[1:,:,:], next_obs[np.newaxis,:,:])
            self.last_adv_frames = row_concat(self.last_adv_frames[1:,:,:], next_obs[np.newaxis,:,:])

            if self.adversary_fn is not None:
                # Compute adversarial perturbation for next_obs
                if self.persistent_adv:
                    # Compute perturbation where last (N_FRAMES-1) frames are adversarial
                    next_adv_obs = self.adversary_fn(self.last_adv_frames)[-1,:,:]
                else:
                    # Compute perturbation where last (N_FRAMES-1) frames are *not* adversarial
                    next_adv_obs = self.adversary_fn(self.last_frames)[-1,:,:]
                self.last_adv_frames[-1,:,:] = next_adv_obs

    def scale_obs(self, obs):
        if self.scale_neg1_1:
            # rescale to [-1,1]
            return (obs / SCALE) * 2.0 - 1.0
        else:
            # rescale to [0,1]
            return obs / SCALE

    def unscale_obs(self, obs):
        assert obs.max() <= 1.0 and obs.min() >= -1.0, "obs is already unscaled"
        if self.scale_neg1_1:
            # rescale from [-1,1] to [0,255]
            return ((obs + 1.0) / 2.0 * SCALE).astype(np.uint8)
        else:
            # rescale from [0,1] to [0,255]
            return (obs * SCALE).astype(np.uint8)

    def preprocess_obs(self, obs):
        # Preprocess Atari frames based on released DQN code from Nature paper:
        #     1) Convert RGB to grayscale (Y in YUV)
        #     2) Rescale image using 'bilinear' method
        obs_y = np.dot(obs, RGB2Y_COEFF)  # Convert RGB to Y
        obs_y = obs_y.astype(np.uint8)  # This is important, otherwise imresize will rescale it to be uint8
        obs_y = imresize(obs_y, self.observation_space.shape[1:], interp='bilinear')

        # Another option is to use cv2.resize but that makes items change size
        # and disappear temporarily
        #obs = cv2.resize(obs, self.observation_space.shape[1:][::-1], \
        #                 interpolation=cv2.INTER_LINEAR)
        obs_y = self.scale_obs(obs_y)
        return obs_y

    @overrides
    def step(self, action):
        # next_obs should be Numpy array of shape (210,160,3)
        next_obs, reward, done, info = self.env.step(action)
        self.update_last_frames(next_obs)

        # Sanity check: visualize self.observation, to make sure it's possible
        # for human to play using this simplified input
        #vis_obs = (self.observation + 1.0) / 2.0  # scale to be from [0,1] for visualization
        #for i in range(vis_obs.shape[0]):
        #    cv2.imshow(str(i), vis_obs[i,:,:])
        #cv2.waitKey()

        self._is_terminal = done
        self._reward = reward
        return Step(self.observation, reward, done, **info)

    @overrides
    def reset(self):
        obs = super().reset()  # self.env.reset()
        self.clear_last_frames()
        self.update_last_frames(obs)
        self._is_terminal = False
        self._reward = 0

        return self.observation
