import collections
import os
import sys
import numpy as np
import cv2

from ale_python_interface import ALEInterface
from sandbox.pchen.async_rl.async_rl.envs.base import Env
from sandbox.pchen.async_rl.async_rl.shareable.base import Shareable
from sandbox.pchen.async_rl.async_rl.utils.picklable import Picklable

class AtariEnv(Env,Shareable,Picklable):
    def __init__(self, rom_filename, seed=None, plot=False, n_last_screens=4,
                 frame_skip=4, treat_life_lost_as_terminal=True,
                 crop_or_scale='scale', img_width=84, img_height=84,
                 max_start_nullops=30,
                 record_screen_dir=None,
                 record_ram=False,
                 record_interal_state=True,
                 phase="Train",
                 initial_manual_activation=True,
                 ):
        self.init_params = locals()
        self.init_params.pop('self')
        self.n_last_screens = n_last_screens
        self.treat_life_lost_as_terminal = treat_life_lost_as_terminal
        self.crop_or_scale = crop_or_scale
        self.max_start_nullops = max_start_nullops
        self.img_width = img_width # useless for now
        self.img_height = img_height # useless for now
        self.record_ram = record_ram
        self.record_interal_state = record_interal_state
        self.seed = seed
        self.record_screen_dir = record_screen_dir
        self.plot = plot
        self.rom_filename = rom_filename
        self.frame_skip = frame_skip
        self.phase = phase
        self.initial_manual_activation = initial_manual_activation
        self.unpicklable_list = ["ale"]

        self.configure_ale()
        self.initialize()

    @property
    def game_name(self):
        return os.path.basename(self.rom_filename).split('.')[0]

    def get_img_shape(self):
        return (self.n_last_screens, self.img_width, self.img_height)

    def configure_ale(self):
        self.ale = ALEInterface()
        if self.seed is not None:
            assert self.seed >= 0 and self.seed < 2 ** 16, \
                "ALE's random seed must be represented by unsigned int"
        else:
            # Use numpy's random state
            self.seed = np.random.randint(0, 2 ** 16)
        # beware that all strings passed to ale must be bytes
        self.ale.setInt(b'random_seed', self.seed)
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.setBool(b'color_averaging', False)
        if self.record_screen_dir is not None:
            self.ale.setString(b'record_screen_dir', str.encode(self.record_screen_dir))
        if self.plot:
            self.prepare_plot()
        else:
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

    def prepare_sharing(self):
        self.share_params = dict()

    def process_copy(self):
        """
        Simply create an env with the same initialization.
        """
        new_env = AtariEnv(**self.init_params)
        return new_env


    def current_screen(self):
        """
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
    def extra_infos(self):
        extra_infos = {}
        if self.record_ram:
            extra_infos["ram_state"] = np.copy(self.ram_state)
        if self.record_interal_state:
            extra_infos["internal_state"] = self.ale.cloneState()
        return extra_infos

    @property
    def state(self):
        assert len(self.last_screens) == self.n_last_screens
        return list(self.last_screens)

    @property
    def is_terminal(self):
        if self.treat_life_lost_as_terminal:
            return self.lives_lost or self.ale.game_over()
        else:
            return self.ale.game_over()

    @property
    def reward(self):
        return self._reward

    @property
    def number_of_actions(self):
        return len(self.legal_actions)

    def receive_action(self, action):
        if self.phase == "Train":
            self.treat_life_lost_as_terminal = True
        else:
            self.treat_life_lost_as_terminal = False

        assert not self.is_terminal

        # Accumulate rewards for repeating this action
        rewards = []
        for i in range(self.frame_skip):

            # Last screeen must be stored before executing the 4th action
            if i == (self.frame_skip - 1):
                self.last_raw_screen = self.ale.getScreenRGB()

            # note that legal actions are integers, but not necessarily consecutive
            # for example, Breakout's minimal action set is [0,1,3,4]
            rewards.append(self.ale.act(self.legal_actions[action]))

            # Check if lives are lost
            if self.lives > self.ale.lives():
                self.lives_lost = True
            else:
                self.lives_lost = False
            self.lives = self.ale.lives()

            if self.is_terminal:
                break

        # We must have last screen here unless it's terminal
        if not self.is_terminal:
            self.last_screens.append(self.current_screen())
            if self.record_ram:
                self.ale.getRAM(self.ram_state)

        self._reward = sum(rewards)

        return self._reward

    def initialize(self):
        """
        Called when starting a new episode, but not a new game.
        """
        # do not directly reset the game, because lost-of-life is not regarded as gameover by default
        if self.ale.game_over():
            self.ale.reset_game()

        # insert randomness to the initial state
        # for example, in Frostbite, the agent waits for a random number of time steps, and the temperature will decrease
        if self.max_start_nullops > 0:
            n_nullops = np.random.randint(0, self.max_start_nullops + 1)
            for _ in range(n_nullops):
                self.ale.act(0)

        # sometimes the game doesn't start at all without performing actions other than noop,
        # but the agent may not learn this at all
        if self.initial_manual_activation:
            if self.game_name == "breakout":
                self.ale.act(3) # fire button, which does nothing but start the game

        self._reward = 0

        self.last_raw_screen = self.ale.getScreenRGB()

        self.last_screens = collections.deque(
            [np.zeros((84, 84), dtype=np.uint8)] * (self.n_last_screens - 1) +
            [self.current_screen()],
            maxlen=self.n_last_screens)

        self.lives_lost = False
        self.lives = self.ale.lives()

    def get_snapshot(self):
        return dict(init_params=self.init_params)

    def update_params(self, global_vars, training_args):
        pass

    def __setstate__(self,state):
        super().__setstate__(state)
        self.configure_ale()
