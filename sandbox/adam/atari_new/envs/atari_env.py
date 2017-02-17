"""
The NEW one that should be optimized.
"""


import numpy as np
import os
import sys
# import gym
# from gym import error, spaces
# from gym import utils
# from gym.utils import seeding

# try:
#     import atari_py
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you can install Atari dependencies by running 'pip install gym[atari]'.)".format(e))

# import logging
# logger = logging.getLogger(__name__)

import atari_py
import cv2
from rllab import spaces
from sandbox.adam.atari_new.spaces.discrete import Discrete as Discrete_n
from rllab.core.serializable import Serializable
from rllab.envs.base import Env

HEIGHT = 84  # Image resize
WIDTH = 84

# BGR2Y_COEFF = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)  # Y = np.dot(bgr, BGR2Y_COEFF)
# CV2_BGR2GRAY = np.array([0.114, 0.587, 0.299], dtype=np.float32)


class AtariEnv(Env, Serializable):

    def __init__(self, game="pong", obs_type="ram", frame_skip=4, two_frame_max=False):
        Serializable.quick_init(self, locals())
        assert obs_type in ("ram", "image")
        game_path = atari_py.get_game_path(game)
        if not os.path.exists(game_path):
            raise IOError("You asked for game %s but path %s does not exist" % (game, game_path))
        self.ale = atari_py.ALEInterface()
        self.ale.loadROM(game_path)
        self._obs_type = obs_type
        self._action_set = self.ale.getMinimalActionSet()
        # self._action_space = spaces.Discrete(len(self._action_set))
        self._action_space = Discrete_n(len(self._action_set))
        self.step = self._step  # default definition
        if self._obs_type == "ram":
            self._observation_space = spaces.Box(low=-1.0 * np.ones(128), high=np.ones(128))
            self._get_obs = self._get_obs_ram
        elif self._obs_type == "image":
            self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(HEIGHT, WIDTH))
            self._get_obs = self._get_obs_image
            if two_frame_max:
                self.step = self._step_two_frame_max
        self.frame_skip = frame_skip
        self.two_frame_max = two_frame_max

    def _step_two_frame_max(self, action):
        """ self.step defined in __init__, this is one option """
        reward = 0.0
        a = self._action_set[action]
        for _ in range(self.frame_skip - 1):
            reward += self.ale.act(a)
        frame_1 = self.ale.getScreenGrayscale()
        reward += self.ale.act(a)
        frame_2 = self.ale.getScreenGrayscale()
        obs = self._process_two_frames(frame_1, frame_2)
        # To render with effects of two frame max, following 2 lines:
        # cv2.imshow("frame_max", (obs + 1) / 2)
        # cv2.waitKey(25)
        return obs, reward, self.ale.game_over(), {}

    def _step(self, action):
        """ self.step defined in __init__, this is one option """
        reward = 0.0
        a = self._action_set[action]
        for _ in range(self.frame_skip):
            reward += self.ale.act(a)
        obs = self._get_obs()
        # self.render()
        return obs, reward, self.ale.game_over(), {}

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _process_two_frames(self, frame_1, frame_2):
        frame_max = np.maximum(frame_1, frame_2)  # highest pixel value in either
        return self._process_frame(frame_max)

    def _process_frame(self, frame):
        # resize faster with float32 input, depending on interpolator
        resized = cv2.resize(frame.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        # self._compare_resize_interps(frame)  # optional diagnostic
        return resized / 255 * 2 - 1  # [-1, 1]

    def _get_obs_image(self):
        frame = self.ale.getScreenGrayscale()
        return self._process_frame(frame)

    def _compare_resize_interps(self, frame):
        interpolations = dict(
            area=cv2.INTER_AREA,
            linear=cv2.INTER_LINEAR,
            lanczos=cv2.INTER_LANCZOS4,
            cubic=cv2.INTER_CUBIC,
            nearest=cv2.INTER_NEAREST
        )
        imgs = dict()
        for k, v in interpolations.items():
            imgs[k] = cv2.resize(frame.astype(np.float32), (WIDTH, HEIGHT), interpolation=v)
        for k, v in imgs.items():
            cv2.imshow(k, v / 255)
            cv2.waitKey(25)

    def _get_rgb_frame(self):
        """ Currently only used in two-frame test """
        (screen_width, screen_height) = self.ale.getScreenDims()
        # ALE seg-faults when ale.getScreenRGB is called without providing a
        # buffer variable, or when providing a buffer variable with 3 channels.
        # It works properly with 4 channels in the buffer (32-bits per pixel),
        # but later the screen data must be copied to make a contiguous variable
        # with 3 channels (the 4th channel is left at zero). This copying
        # actually takes a significant amount of time.  (Passing the slice
        # arr[:, :, 0:3] into cv2.resize results in the same execution time as
        # passing arr[:, :, 0:3].copy().)  Instead, use ale.getScreenGrayscale,
        # which looks the same as using cv2 to convert BGR2GRAY.
        arr = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)  # Not sure why the 4 here instead of 3?
        self.ale.getScreenRGB(arr)

        # The returned values are in 32-bit chunks. How to unpack them into
        # 8-bit values depend on the endianness of the system
        if sys.byteorder == 'little':  # the layout is BGRA
            arr = arr[:, :, 0:3].copy()
        else:  # the layout is ARGB (I actually did not test this.
            # Need to verify on a big-endian machine)
            arr = arr[:, :, 2::-1]
        # img = arr[1:-1, :, :]  # Crops one row??
        return arr

    def _process_rgb_frame(self, frame):
        """ Not used, currently """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img_grey = frame.dot(CV2_BGR2GRAY)  # this is slower than cv2.cvtColor
        resized = cv2.resize(frame.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        return resized / 255 * 2 - 1  # normalize to [-1, 1]

    def _get_obs_ram(self):
        ram = self.ale.getRAM().astype(np.float32)
        return ram / 255 * 2 - 1  # normalize to [-1, 1]

    def reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def render(self):
        """ Does NOT include two-frame maximizing """
        img = self._get_obs_image()  # img: [-1, 1]
        cv2.imshow("atarigame", (img + 1) / 2)  # img: [-1, 1] -> [0, 1]
        cv2.waitKey(10)

    def step_two_frame_test(self, action):
        """
        Use to see whether games need two-frame pixel maximizer:
        env.step = env.step_two_frame_test
        """
        reward = 0.0
        a = self._action_set[action]
        for _ in range(self.frame_skip - 1):
            reward += self.ale.act(a)
        frame_1 = self.ale.getScreenGrayscale()
        rgb_1 = self._get_rgb_frame()
        cv2.imshow("rgb_1", rgb_1)
        cv2.waitKey(25)
        reward += self.ale.act(a)
        frame_2 = self.ale.getScreenGrayscale()
        obs = self._process_two_frames(frame_1, frame_2)
        rgb_2 = self._get_rgb_frame()
        cv2.imshow("rgb_max", np.maximum(rgb_1, rgb_2))
        # To render with effects of two frame max, following 2 lines:
        cv2.imshow("gray_max", (obs + 1) / 2)
        cv2.waitKey(25)
        return obs, reward, self.ale.game_over(), {}

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

