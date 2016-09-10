import collections
import os
import sys
import numpy as np
import cv2
import copy
import atari_py

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from sandbox.haoran.ale_python_interface import ALEInterface

class AtariEnv(Env,Serializable):
    def __init__(self,
            game,
            seed=None,
            plot=False, # live demo
            max_start_nullops=0,
            obs_type="image",
            record_image=True, # image for training and counting
            record_rgb_image=False, # for visualization and debugging
            record_ram=False,
            record_internal_state=True,
            resetter=None,
        ):
        """
        plot: not compatible with rllab yet
        """
        Serializable.quick_init(self,locals())
        assert not plot
        self.rom_filename = atari_py.get_game_path(game)
        self.seed = seed
        self.plot = plot
        self.max_start_nullops = max_start_nullops
        self.obs_type = obs_type
        self.record_image = record_image
        self.record_rgb_image = record_rgb_image
        self.record_ram = record_ram
        self.record_internal_state = record_internal_state
        self.resetter = resetter
        if resetter is not None:
            assert max_start_nullops == 0 # doing nothing when reset to a non-initial state can be dangerous in Montezuma's Revenge

        self.frame_skip = 4
        self.n_last_screens = 4
        self.crop_or_scale = 'scale'

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
        self.ale.setBool(b'color_averaging', False)
        if self.plot:
            self.prepare_plot()
        else:
            if not os.path.exists(self.rom_filename):
                raise IOError("You asked for game %s but path %s does not exist" % (self.game_name, self.rom_filename))
            self.ale.loadROM(str.encode(self.rom_filename))

        assert self.ale.getFrameNumber() == 0

        self.legal_actions = self.ale.getMinimalActionSet()
        if self.record_ram:
            self.ram_state = np.zeros(self.ale.getRAMSize(), dtype=np.uint8)

    def prepare_plot(self,display="0.0"):
        os.environ["DISPLAY"] = display
        # SDL settings below are from the ALE python example
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            self.ale.setBool(b'sound', False)  # Sound doesn't work on OSX
        elif sys.platform.startswith('linux'):
            self.ale.setBool(b'sound', True)
        self.ale.setBool(b'display_screen', True)
        self.ale.loadROM(str.encode(self.rom_filename))

    def set_seed(self,seed):
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
        elif self.crop_or_scale == 'scale':
            img = cv2.resize(img, (84, 84),
                             interpolation=cv2.INTER_LINEAR)
        else:
            raise RuntimeError('crop_or_scale must be either crop or scale')
        assert img.shape == (84, 84)
        return img

    @property
    def game_name(self):
        return os.path.basename(self.rom_filename).split('.')[0]

    @property
    def observation(self):
        if self.obs_type == "image":
            assert len(self.last_screens) == self.n_last_screens
            imgs = np.asarray(list(self.last_screens))
            imgs = (imgs / 255.0) * 2.0 - 1.0 # rescale to [-1,1]
            return imgs
        elif self.obs_type == "ram":
            ram = (self.ram_state / 255.0) * 2.0 - 1.0 # rescale to [-1,1]
            return ram
        else:
            raise NotImplementedError

    @property
    def observation_space(self):
        if self.obs_type == "ram":
            return spaces.Box(low=-1, high=1, shape=(self.ale.getRAMSize(),))#np.zeros(128), high=np.ones(128))# + 255)
        elif self.obs_type == "image":
            return spaces.Box(low=-1, high=1, shape=(self.n_last_screens, 84,84))

    @property
    def action_space(self):
        return spaces.Discrete(len(self.legal_actions))

    @property
    def is_terminal(self):
        return self.ale.game_over()

    @property
    def reward(self):
        return self._reward

    @property
    def env_info(self):
        env_info = {}
        if self.record_ram and self.obs_type != "ram":
            env_info["ram_states"] = np.copy(self.ram_state)

        if self.record_image and self.obs_type != "image":
            env_info["images"] = np.copy(np.asarray(list(self.last_screens)))

        if self.record_rgb_image:
            env_info["rgb_images"] = self.ale.getScreenRGB()

        if self.record_internal_state:
            env_info["internal_states"] = self.ale.cloneState()

        env_info["is_terminals"] = self.is_terminal
        env_info["ale_ids"] = hex(id(self.ale))
        env_info["use_default_reset"] = self.cur_path_use_default_reset

        return env_info


    def step(self, action):
        cur_env_info = copy.deepcopy(self.env_info)
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
            if self.is_terminal:
                break
        self._reward = sum(rewards)

        # Record next step env info
        if not self.is_terminal:
            if self.record_image:
                self.last_screens.append(self.current_screen())
            if self.record_ram:
                self.ale.getRAM(self.ram_state)

        # cur_obs, cur_reward, next_state_is_terminal, cur_env_info
        return self.observation, self.reward, self.is_terminal, cur_env_info

    def reset(self):
        if self.resetter is None:
            self.ale.reset_game()
            self.cur_path_use_default_reset = True
        else:
            self.cur_path_use_default_reset = self.resetter.reset(self)

        # insert randomness to the initial state
        # for example, in Frostbite, the agent waits for a random number of time steps, and the temperature will decrease
        if self.max_start_nullops > 0:
            n_nullops = np.random.randint(0, self.max_start_nullops + 1)
            for _ in range(n_nullops):
                self.ale.act(0)

        self._reward = 0

        self.last_raw_screen = self.ale.getScreenRGB()

        self.last_screens = collections.deque(
            [np.zeros((84, 84), dtype=np.uint8)] * (self.n_last_screens - 1) +
            [self.current_screen()],
            maxlen=self.n_last_screens)
        return self.observation

    def render(self,return_array=False):
        img = self.last_screens[-1]
        cv2.imshow(self.game_name, img)
        cv2.waitKey(10)
        if return_array:
            return img

    def get_param_values(self):
        params = dict()
        if self.resetter is not None:
            params["resetter_params"] = self.resetter.get_param_values()
        return params


    def set_param_values(self,params):
        if self.resetter is not None:
            self.resetter.set_param_values(params["resetter_params"])
