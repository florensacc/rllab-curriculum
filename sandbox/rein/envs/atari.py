import numpy as np
import os
import sys
import atari_py

import logging

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
import scipy

logger = logging.getLogger(__name__)


class NoCV2(object):
    """
    A class to be used in case cv2 isn't installed
    """

    def __getattr__(self, *args, **kwargs):
        raise ("""The cv2 (that is, opencv) library is not installed, but a function was just called that requires it. Please install cv2 or stick to methods that don't use cv2.

(HINT: Unfortunately, cv2 can't just be installed via pip. On OSX, you'll need to do something like: conda install opencvbrew install opencv.

On Ubuntu you can follow https://help.ubuntu.com/community/OpenCV to install OpenCV.
If you would like to use virutalenv also include these flags when you call cmake:
CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/local/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages

If you don't want to install cv2, note that Atari RAM environments don't require cv2, so long as you don't try to render.)""")


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return gray


try:
    import cv2
except ImportError:
    logger.warn(
        "atari.py: Couldn't import opencv. Atari image environments will raise an error (RAM ones will still work).")
    cv2 = NoCV2()

IMG_WH = (42, 52)  # width, height of images


def to_rgb(ale):
    (screen_width, screen_height) = ale.getScreenDims()
    arr = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    ale.getScreenRGB(arr)
    # The returned values are in 32-bit chunks. How to unpack them into
    # 8-bit values depend on the endianness of the system
    if sys.byteorder == 'little':  # the layout is BGRA
        arr = arr[:, :, 0:3].copy()  # (0, 1, 2) <- (2, 1, 0)
    else:  # the layout is ARGB (I actually did not test this.
        # Need to verify on a big-endian machine)
        arr = arr[:, :, 2:-1:-1]
    img = arr
    return img


def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size), dtype=np.uint8)
    ale.getRAM(ram)


class AtariEnvX(Env, Serializable):
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
        self.frame_skip = frame_skip

    def step(self, a):
        reward = 0.0
        action = self._action_set[a]
        for _ in range(self.frame_skip):
            reward += self.ale.act(action)
        ob = self._get_obs()
        return ob, reward, self.ale.game_over(), {}

    @property
    def action_space(self):
        return spaces.Discrete(len(self._action_set))

    @property
    def observation_space(self):
        if self._obs_type == "ram":
            return spaces.Box(low=-1, high=1, shape=(128,))  # np.zeros(128), high=np.ones(128))# + 255)
        elif self._obs_type == "image":
            return spaces.Box(low=-1, high=1, shape=(1,) + IMG_WH[::-1])

    def _get_image(self):
        return to_rgb(self.ale)

    def _get_ram(self):
        return to_ram(self.ale)

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == "ram":
            ram = self._get_ram()
            # scale to [0, 1]
            ram = ram / 255.0
            # scale to [-1, 1]
            ram = ram * 2.0 - 1.0
            return ram
        elif self._obs_type == "image":
            next_obs = self._get_image()[1:-1, :, :]
            next_obs = scipy.misc.imresize(
                next_obs, (IMG_WH[1], IMG_WH[0], 3), interp='bicubic', mode=None)
            next_obs = rgb2gray(next_obs)
            next_obs = next_obs / 256.
            next_obs = next_obs[np.newaxis, :, :]
            return next_obs

    # return: (states, observations)
    def reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def render(self, return_array=False):
        img = self._get_image()
        cv2.imshow("atarigame", img)
        cv2.waitKey(10)
        if return_array: return img
