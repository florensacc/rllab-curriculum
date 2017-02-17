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
from rllab.core.serializable import Serializable
from rllab.envs.base import Env

HEIGHT = 84  # Image resize
WIDTH = 84

# RGB2Y_COEFF = np.array([0.2126, 0.7152, 0.0722])  # Y = np.dot(rgb, RGB2Y_COEFF)
# BGR2Y_COEFF = np.array([0.0722, 0.7152, 0.2126])  # Y = np.dot(grb, BGR2Y_COEFF)


class AtariEnv(Env, Serializable):

    def __init__(self, game="pong", obs_type="ram", frame_skip=4):
        Serializable.quick_init(self, locals())
        assert obs_type in ("ram", "image")
        game_path = atari_py.get_game_path(game)
        if not os.path.exists(game_path):
            raise IOError("You asked for game %s but path %s does not exist" % (game, game_path))
        self.ale = atari_py.ALEInterface()
        self.ale.loadROM(game_path)
        self._obs_type = obs_type
        self._action_set = self.ale.getMinimalActionSet()
        self._action_space = spaces.Discrete(len(self._action_set))
        if self._obs_type == "ram":
            self._observation_space = spaces.Box(low=-1.0 * np.ones(128), high=np.ones(128))
            self._get_obs = self._get_obs_ram
        elif self._obs_type == "image":
            self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(HEIGHT, WIDTH))
            self._get_obs = self._get_obs_image

        self.frame_skip = frame_skip

    def step(self, action):
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

    def _get_obs_image(self):
        (screen_width, screen_height) = self.ale.getScreenDims()
        arr = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)  # Not sure why the 4 here instead of 3?
        self.ale.getScreenRGB(arr)
        # The returned values are in 32-bit chunks. How to unpack them into
        # 8-bit values depend on the endianness of the system
        if sys.byteorder == 'little':  # the layout is BGRA
            arr = arr[:, :, 0:3].copy()  # (0, 1, 2) <- (2, 1, 0)
        else:  # the layout is ARGB (I actually did not test this.
            # Need to verify on a big-endian machine)
            arr = arr[:, :, 2:-1:-1]
        img = arr[1:-1, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img = img / 255.0 * 2.0 - 1.0  # normalize to [-1, 1]
        return img

    def _get_obs_ram(self):
        ram = self.ale.getRAM()
        ram = ram / 255.0 * 2.0 - 1.0  # normalize to [-1, 1]
        return ram

    def reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def render(self, return_array=False):
        img = self._get_obs_image()
        img = (img + 1.0) / 2.0 * 255.0
        img = img.astype(np.uint8)
        cv2.imshow("atarigame", img)
        cv2.waitKey(10)
        if return_array:
            return img

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
