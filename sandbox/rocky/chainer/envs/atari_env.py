import collections
import os
import sys
import numpy as np
import cv2
import copy
import atari_py

from rllab import config
from rllab.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from sandbox.rocky.chainer.envs.ale_python_interface import ALEInterface
from rllab.spaces import Box


class AtariEnv(Env, Serializable):
    def __init__(self,
                 game,
                 seed=None,
                 max_start_nullops=0,
                 img_width=84,
                 img_height=84,
                 crop_or_scale='scale',
                 obs_type="image",
                 avoid_life_lost=False,
                 n_last_rams=1,
                 n_last_screens=4,
                 frame_skip=4,
                 rom_filename="",
                 ):
        Serializable.quick_init(self, locals())
        if rom_filename == "":
            self.rom_filename = atari_py.get_game_path(game)
        else:
            self.rom_filename = rom_filename
        self.seed = seed
        self.max_start_nullops = max_start_nullops
        self.obs_type = obs_type
        self.crop_or_scale = crop_or_scale
        self.img_width = img_width
        self.img_height = img_height
        self._prior_reward = 0
        self.frame_skip = frame_skip
        self.n_last_screens = n_last_screens
        self.n_last_rams = n_last_rams
        self.avoid_life_lost = avoid_life_lost

        self.configure_ale()
        self.reset()

    def configure_ale(self):
        self.ale = ALEInterface()

        # set seed
        if self.seed is not None:
            assert self.seed >= 0 and self.seed < 2 ** 16, \
                "ALE's random seed must be represented by unsigned int"
        else:
            self.seed = np.random.randint(0, 2 ** 16)
        self.set_seed(self.seed)

        self.ale.setFloat(b'repeat_action_probability', 0.0)
        if self.game_name == "venture":
            # without color averaging, the agent and reward items are invisible in Venture
            color_averaging = True
        else:
            color_averaging = False
        self.ale.setBool(b'color_averaging', color_averaging)

        if not os.path.exists(self.rom_filename):
            raise IOError("You asked for game %s but path %s does not exist" % (self.game_name, self.rom_filename))
        self.ale.loadROM(str.encode(self.rom_filename))

        assert self.ale.getFrameNumber() == 0

        # limit the action set to make learning easier
        self.legal_actions = self.ale.getMinimalActionSet()

    def set_seed(self, seed):
        self.ale.setInt(b'random_seed', seed)

    def current_screen(self):
        """
        Can only be called once; subsequent calls should use self.last_screens[-1]
        Very inflexible code.
        Slightly different from usual codes. Max between consecutive frames is taken in the RGB space.
        Then grayscale image is taken by luminance transform.
        Crop 8 pixels from the bottom, and the reset from top.
        """
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
        if self.crop_or_scale == 'crop':
            # Shrink (210, 160) -> (110, 84)
            img = cv2.resize(img, (84, 110),
                             interpolation=cv2.INTER_LINEAR)
            assert img.shape == (110, 84)
            # Crop (110, 84) -> (84, 84)
            unused_height = 110 - 84
            bottom_crop = 8
            top_crop = unused_height - bottom_crop
            img = img[top_crop: 110 - bottom_crop, :]
            img = cv2.resize(img, (self.img_width, self.img_height))
        elif self.crop_or_scale == 'scale':
            img = cv2.resize(img, (self.img_width, self.img_height),
                             interpolation=cv2.INTER_LINEAR)
        else:
            raise RuntimeError('crop_or_scale must be either crop or scale')
        return img

    @property
    def game_name(self):
        return os.path.basename(self.rom_filename).split('.')[0]

    @property
    def observation(self):
        if self.obs_type == "image":
            assert len(self.last_screens) == self.n_last_screens
            imgs = np.asarray(list(self.last_screens))
            imgs = imgs / 255.0
            return imgs
        elif self.obs_type == "ram":
            assert len(self.last_rams) == self.n_last_rams
            rams = np.asarray(list(self.last_rams))
            rams = rams / 255.0
            return rams
        else:
            raise NotImplementedError

    @property
    def observation_space(self):
        if self.obs_type == "ram":
            return Box(
                low=0, high=1,
                shape=(self.n_last_rams, self.ale.getRAMSize())
            )
        elif self.obs_type == "image":
            image_shape = (self.n_last_screens, self.img_width, self.img_height)
            return Box(low=0, high=1, shape=image_shape)
        else:
            raise NotImplementedError

    @property
    def action_space(self):
        return Discrete(len(self.legal_actions))

    @property
    def is_terminal(self):
        if self.avoid_life_lost:
            if self.ale.game_over() or self.lives_lost:
                # print("Terminate due to life loss")
                return True
            else:
                return False
        else:
            if self.ale.game_over():
                # print("Terminate due to gameover")
                return True
            else:
                return False

    @property
    def reward(self):
        return self._reward

    # @property
    # def env_info(self):
    #     env_info = {}
    #
    #     env_info["lives_lost"] = self.lives_lost
    #
    #     return env_info

    def step(self, action):
        # cur_env_info = copy.deepcopy(self.env_info)
        # a legal observation should not be terminal; but to make the program run without interruption, we allow it to reset
        # assert not self.is_terminal
        # if self.is_terminal:
        #     self.reset()

        # Accumulate rewards for repeating this action
        rewards = []
        for i in range(self.frame_skip):

            # Last screeen must be stored before executing the 4th action
            if i == (self.frame_skip - 1):
                self.last_raw_screen = self.ale.getScreenRGB()

            # note that legal actions are integers, but not necessarily consecutive
            # for example, Breakout's minimal action set is [0,1,3,4]
            rewards.append(self.ale.act(self.legal_actions[action]))

            if self.start_lives > self.ale.lives():
                self.lives_lost = True
            else:
                self.lives_lost = False

            if self.is_terminal:
                break
        self._reward = sum(rewards)
        # if self._prior_reward > 0:
        #     cur_env_info["prior_reward"] = self._prior_reward
        #     self._prior_reward = 0
        # else:
        #     cur_env_info["prior_reward"] = 0

        # Record next step env info
        if not self.is_terminal:
            if self.obs_type == "image":
                self.last_screens.append(self.current_screen())
            if self.obs_type == "ram":
                self.last_rams.append(np.copy(self.ale.getRAM()))

        # cur_obs, cur_reward, next_state_is_terminal, cur_env_info
        return self.observation, self.reward, self.is_terminal, dict()

    def reset(self):
        self.ale.reset_game()

        # insert randomness to the initial state
        # for example, in Frostbite, the agent waits for a random number of time steps, and the temperature will decrease
        if self.max_start_nullops > 0:
            n_nullops = np.random.randint(0, self.max_start_nullops + 1)
            for _ in range(n_nullops):
                self.ale.act(0)

        self._reward = 0

        self.last_raw_screen = self.ale.getScreenRGB()

        if self.obs_type == "image":
            self.last_screens = collections.deque(
                [np.zeros((self.img_width, self.img_height), dtype=np.uint8)] * (self.n_last_screens - 1) +
                [self.current_screen()],
                maxlen=self.n_last_screens)
        if self.obs_type == "ram":
            self.last_rams = collections.deque(
                [np.zeros(self.ale.getRAMSize(), dtype=np.uint8)] * (self.n_last_rams - 1) + [self.ale.getRAM()],
                maxlen=self.n_last_rams
            )

        self.start_lives = self.ale.lives()
        self.lives_lost = False
        return self.observation

    def render(self, return_array=False):
        img = self.ale.getScreenRGB()
        cv2.imshow(self.game_name, img)
        cv2.waitKey(10)
        if return_array:
            return img
