import numpy as np
import os
import sys
import atari_py
import collections

import logging

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Env

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


try:
    import cv2
except ImportError:
    logger.warn(
        "atari.py: Couldn't import opencv. Atari image environments will raise an error (RAM ones will still work).")
    cv2 = NoCV2()

IMG_WH = (40, 52)  # width, height of images


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
    return ram


class AtariEnv(Env, Serializable):
    def __init__(self, game="pong",
            obs_type="ram",
            frame_skip=4,
            death_ends_episode=False,
            death_penalty=0,
            n_last_screens=0,
            image_resize_method="scale",
        ):
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
        self.death_ends_episode = death_ends_episode
        self.death_penalty = death_penalty
        self.n_last_screens = n_last_screens
        self.image_resize_method = image_resize_method

        self.last_raw_screen = self._get_image()
        self.last_screens = collections.deque(
            [np.zeros((84, 84), dtype=np.uint8)] * self.n_last_screens,
            maxlen=self.n_last_screens)

    def step(self, a):
        reward = 0.0
        action = self._action_set[a]
        for i in range(self.frame_skip):
            if i == (self.frame_skip - 1):
                self.last_raw_screen = self._get_image()
            reward += self.ale.act(action)
        ob = self._get_obs()
        self.last_screens.append(self.get_current_screen())

        cur_lives = self.ale.lives()
        lose_life = cur_lives < self.start_lives
        if self.death_ends_episode:
            done = self.ale.game_over() or lose_life
        else:
            done = self.ale.game_over()

        if lose_life:
            reward -= self.death_penalty
            logger.log(0,"Death penalty: %f"%(self.death_penalty))

        env_infos = {
            "images": np.asarray(self.last_screens)
        }
        return ob, reward, done, env_infos

    @property
    def action_space(self):
        return spaces.Discrete(len(self._action_set))

    @property
    def observation_space(self):
        if self._obs_type == "ram":
            return spaces.Box(low=-1, high=1, shape=(128,))#np.zeros(128), high=np.ones(128))# + 255)
        elif self._obs_type == "image":
            return spaces.Box(low=-1, high=1, shape=IMG_WH[::-1])

    def get_current_screen(self):
        # Max of two consecutive frames
        assert self.last_raw_screen is not None
        rgb_img = np.maximum(self.ale.getScreenRGB(), self.last_raw_screen)
        # Make sure the last raw screen is used only once
        self.last_raw_screen = None
        assert rgb_img.shape == (210, 160, 3)
        # RGB -> Luminance
        img = rgb_img[:, :, 0] * 0.2126 + rgb_img[:, :, 1] * \
            0.0722 + rgb_img[:, :, 2] * 0.7152
        img = img.astype(np.uint8)
        if img.shape == (250, 160):
            raise RuntimeError("This ROM is for PAL. Please use ROMs for NTSC")
        assert img.shape == (210, 160)
        if self.image_resize_method == 'crop':
            # Shrink (210, 160) -> (110, 84)
            img = cv2.resize(img, (84, 110),
                             interpolation=cv2.INTER_LINEAR)
            assert img.shape == (110, 84)
            # Crop (110, 84) -> (84, 84)
            unused_height = 110 - 84
            bottom_crop = 8
            top_crop = unused_height - bottom_crop
            img = img[top_crop: 110 - bottom_crop, :]
        elif self.image_resize_method == 'scale':
            img = cv2.resize(img, (84, 84),
                             interpolation=cv2.INTER_LINEAR)
        else:
            raise RuntimeError('crop_or_scale must be either crop or scale')
        assert img.shape == (84, 84)
        return img

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
            img = self._get_image()[1:-1, :, :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_WH, interpolation=cv2.INTER_AREA)
            # scale to [0, 1]
            img = img / 255.0
            # scale to [-1, 1]
            img = img * 2.0 - 1.0
            return img

    # return: (states, observations)
    def reset(self):
        self.ale.reset_game()
        self.start_lives = self.ale.lives()
        self.last_screens = collections.deque(
            [np.zeros((84, 84), dtype=np.uint8)] * self.n_last_screens,
            maxlen=self.n_last_screens)
        return self._get_obs()

    def render(self, return_array=False):
        img = self._get_image()
        cv2.imshow("atarigame", img)
        cv2.waitKey(10)
        if return_array: return img
