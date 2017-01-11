import collections
import os
import sys
import numpy as np
import cv2
import copy
import atari_py
from scipy.misc import imresize

from gym.utils import seeding
from rllab import config
from rllab.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from sandbox.haoran.ale_python_interface import ALEInterface

class AtariEnvMinimal(Env,Serializable):
    def __init__(self,
            game,
            seed=None,
            plot=False, # live demo
            max_start_nullops=0,
            img_width=84,
            img_height=84,
            resize_with_scipy=False,
            obs_type="image",
            record_image=True, # image for training and counting
            record_rgb_image=False, # for visualization and debugging
            record_ram=False,
            record_internal_state=True,
            n_last_rams=1,
            n_last_screens=4,
            frame_skip=4,
            legal_actions=[],
            rom_filename="",
            correct_luminance=True,
            recorded_rgb_image_scale=1.0,
        ):
        """
        plot: not compatible with rllab yet
        """
        Serializable.quick_init(self,locals())
        assert not plot
        if rom_filename == "":
            self.rom_filename = atari_py.get_game_path(game)
        else:
            self.rom_filename = rom_filename
        self.seed = seed
        self.plot = plot
        self.max_start_nullops = max_start_nullops
        self.obs_type = obs_type
        self.record_image = record_image
        self.record_rgb_image = record_rgb_image
        self.recorded_rgb_image_scale = recorded_rgb_image_scale
        self.record_ram = record_ram
        self.record_internal_state = record_internal_state
        self.resize_with_scipy = resize_with_scipy
        self.img_width = img_width
        self.img_height = img_height
        self._prior_reward = 0
        self.frame_skip = frame_skip
        self.n_last_screens = n_last_screens
        self.n_last_rams = n_last_rams
        self.legal_actions = legal_actions
        self.correct_luminance = correct_luminance
        if not correct_luminance:
            print("""
            ################################################
            ################################################
            ################################################
            ################################################

            Warning: not using the correct luminance

            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            """)

        # adversary_fn - function handle for adversarial perturbation of observation
        self.adversary_fn = None

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

        if self.plot:
            self.prepare_plot()
        else:
            if not os.path.exists(self.rom_filename):
                raise IOError("You asked for game %s but path %s does not exist" % (self.game_name, self.rom_filename))
            self.ale.loadROM(str.encode(self.rom_filename))

        assert self.ale.getFrameNumber() == 0

        # limit the action set to make learning easier
        if len(self.legal_actions) == 0:
            self.legal_actions = self.ale.getMinimalActionSet()

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
        # Copied next two lines from gym.envs.atari.atari_env._seed
        seed2 = seeding.hash_seed(seed + 1) % 2**31
        self.ale.setInt(b'random_seed', seed2)
        #self.ale.setInt(b'random_seed', seed)

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

        # NOTE: Testing whether this is necessary
        #rgb_img = np.maximum(self.ale.getScreenRGB(), self.last_raw_screen)
        rgb_img = self.ale.getScreenRGB()

        # Make sure the last raw screen is used only once
        self.last_raw_screen = None
        assert rgb_img.shape == (210, 160, 3)
        # RGB -> Luminance
        if self.correct_luminance:
            img = rgb_img[:, :, 0] * 0.2126 + rgb_img[:, :, 1] * \
                0.0722 + rgb_img[:, :, 2] * 0.7152
        else:
            img = rgb_img[:, :, 0] * 0.2126 + rgb_img[:, :, 1] * \
                0.7152 + rgb_img[:, :, 2] * 0.0722
        img = img.astype(np.uint8)
        if img.shape == (250, 160):
            raise RuntimeError("This ROM is for PAL. Please use ROMs for NTSC")
        assert img.shape == (210, 160)

        if self.resize_with_scipy:
            img = imresize(img, (self.img_width, self.img_height), interp='bilinear')
        else:
            img = cv2.resize(img, (self.img_width, self.img_height),
                             interpolation=cv2.INTER_LINEAR)
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
            if self.adversary_fn is not None:
                # Adversarially perturb input
                imgs = self.adversary_fn(imgs)
            return imgs
        elif self.obs_type == "ram":
            assert len(self.last_rams) == self.n_last_rams
            rams = np.asarray(list(self.last_rams))
            rams = (rams / 255.0) * 2.0 - 1.0
            return rams
        else:
            raise NotImplementedError

    def set_adversary_fn(self, fn_handle):
        self.adversary_fn = fn_handle

    @property
    def observation_space(self):
        if config.USE_TF:
            from sandbox.rocky.tf.spaces.box import Box
        else:
            from rllab.spaces import Box

        if self.obs_type == "ram":
            return Box(low=-1, high=1,
                shape=(self.n_last_rams, self.ale.getRAMSize())
            ) #np.zeros(128), high=np.ones(128))# + 255)
        elif self.obs_type == "image":
            if config.USE_TF:
                image_shape = (self.img_width, self.img_height, self.n_last_screens)
            else:
                image_shape = (self.n_last_screens, self.img_width, self.img_height)
            return Box(low=-1, high=1, shape=image_shape)
            # see sandbox.haoran.tf.core.layers.BaseConvLayer for a reason why channel is at the last dimension
        else:
            raise NotImplementedError

    @property
    def action_space(self):
        return Discrete(len(self.legal_actions))

    @property
    def is_terminal(self):
        if self.ale.game_over():
            # print("Terminate due to gameover")
            return True
        else:
            return False

    @property
    def reward(self):
        return self._reward

    @property
    def env_info(self):
        env_info = {}
        # if self.record_ram and self.obs_type != "ram":
        if self.record_ram:
            ram = np.copy(self.ale.getRAM())
            ram = ram.reshape((1,len(ram),1)) # make it like an image
            env_info["ram_states"] = ram

        if self.record_image and self.obs_type != "image":
            env_info["images"] = np.copy(np.asarray(list(self.last_screens)))

        if self.record_rgb_image:
            rgb_img = self.ale.getScreenRGB()
            scale = self.recorded_rgb_image_scale
            if abs(scale-1.0) > 1e-4:
                rgb_img = cv2.resize(rgb_img, dsize=(0,0),fx=scale,fy=scale)
            env_info["rgb_images"] = rgb_img

        if self.record_internal_state:
            env_info["internal_states"] = self.ale.cloneState()

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
        if self._prior_reward > 0:
            cur_env_info["prior_reward"] = self._prior_reward
            self._prior_reward = 0
        else:
            cur_env_info["prior_reward"] = 0

        # Record next step env info
        if not self.is_terminal:
            if self.record_image or self.obs_type == "image":
                self.last_screens.append(self.current_screen())
            if self.record_ram or self.obs_type == "ram":
                self.last_rams.append(np.copy(self.ale.getRAM()))

        # cur_obs, cur_reward, next_state_is_terminal, cur_env_info
        return self.observation, self.reward, self.is_terminal, cur_env_info

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

        if self.obs_type == "image" or self.record_image:
            self.last_screens = collections.deque(
                [np.zeros((self.img_width, self.img_height), dtype=np.uint8)] * (self.n_last_screens - 1) +
                [self.current_screen()],
                maxlen=self.n_last_screens)
        if self.obs_type == "ram" or self.record_ram:
            self.last_rams = collections.deque(
                [np.zeros(self.ale.getRAMSize(), dtype=np.uint8)] * (self.n_last_rams - 1) + [self.ale.getRAM()],
                maxlen=self.n_last_rams
            )

        return self.observation

    def render(self,return_array=False):
        img = self.ale.getScreenRGB()
        cv2.imshow(self.game_name, img)
        cv2.waitKey(10)
        if return_array:
            return img
