import collections

import itertools

import numba
import numpy as np

from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces import Discrete
from rllab.spaces.box import Box
from cached_property import cached_property
import pyximport
import os
import atari_py
import cv2
import sys

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension


def new_get_distutils_extension(modname, pyxfilename, language_level=None):
    atari_dir = os.path.dirname(os.path.realpath(atari_py.__file__))
    extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language = 'c++'
    extension_mod.library_dirs = [os.path.join(atari_dir, "ale_interface/build")]
    extension_mod.runtime_library_dirs = [os.path.join(atari_dir, "ale_interface/build")]
    extension_mod.include_dirs = [os.path.join(atari_dir, "ale_interface/src"), np.get_include()]
    extension_mod.extra_compile_args = ['-fopenmp']
    extension_mod.extra_link_args = ['-fopenmp']
    extension_mod.libraries = ["ale"]
    return extension_mod, dict(setup_args, script_args=['--verbose'])


pyximport.pyximport.get_distutils_extension = new_get_distutils_extension
pyximport.install()

from sandbox.rocky.neural_learner.envs import par_ale

rgb_palette = np.asarray([
    0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
    0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
    0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
    0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
    0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
    0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
    0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
    0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
    0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
    0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
    0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
    0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
    0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
    0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
    0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
    0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
    0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
    0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
    0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
    0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
    0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
    0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
    0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
    0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
    0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
    0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
    0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
    0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
    0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
    0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
    0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
    0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
], dtype=np.int32).view(np.uint8).reshape((256, 4))

if sys.byteorder == 'big':
    rgb_palette = rgb_palette[:, ::-1]
elif sys.byteorder == 'little':
    pass
else:
    raise NotImplementedError

rgb_palette = np.cast['uint8'](rgb_palette[:, :-1]) / 255.0

gray_palette = rgb_palette[:, 0] * 0.2126 + rgb_palette[:, 1] * 0.0722 + rgb_palette[:, 2] * 0.7152


@numba.vectorize(["float32(uint8)", "float64(uint8)"])
def decode_gray(x):
    return gray_palette[x]


class NaiveCircularBuffer(object):
    def __init__(self, batch_size, buffer_size, data_shape, dtype):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.data_shape = data_shape
        self.dtype = dtype
        self.buffer = []
        for _ in range(batch_size):
            self.buffer.append(collections.deque(
                [np.zeros(data_shape, dtype=dtype)] * buffer_size,
                maxlen=buffer_size
            ))
            # np.zeros((size,) + data_shape, dtype=dtype)
            # self.buffer_head = np.zeros((size,), dtype=np.int)

    def reset(self, mask=None):
        if mask is None:
            mask = [True] * self.batch_size
        for i, mask_i in enumerate(mask):
            if mask_i:
                self.buffer[i].clear()
                self.buffer[i].extend([np.zeros(self.data_shape, dtype=self.dtype)] * self.buffer_size)

    def push(self, data, mask=None):
        if mask is None:
            mask = [True] * self.batch_size
        inc = 0
        for i, mask_i in zip(itertools.count(), mask):
            if mask_i:
                self.buffer[i].append(data[inc])
                inc += 1

    def last(self, n_last):
        # This should return the content of the buffer within head-n_last (inclusive) ... head (exclusive)
        # How to vectorize this?
        assert n_last <= self.buffer_size
        ret = []
        for buf in self.buffer:
            ret.append(list(buf)[-n_last:])
        return np.asarray(ret)


class CircularBuffer(object):
    def __init__(self, size, data_shape, dtype):
        self.size = size
        self.buffer = np.zeros((size,) + data_shape, dtype=dtype)
        self.buffer_head = np.zeros((size,), dtype=np.int)

    def reset(self, mask=None):
        if mask is None:
            self.buffer[:] = 0
            self.buffer_head[:] = 0
        else:
            mask = np.cast['bool'](mask)
            self.buffer[mask] = 0
            self.buffer_head[mask] = 0

    def push(self, data, mask=None):
        if mask is None:
            self.buffer[np.arange(self.size), :, :, self.buffer_head] = data
            self.buffer_head = (self.buffer_head + 1) % self.size
        else:
            mask = np.cast['bool'](mask)
            self.buffer[np.arange(self.size)[mask], :, :, self.buffer_head[mask]] = data
            self.buffer_head[mask] = (self.buffer_head[mask] + 1) % self.size

    def last(self, n_last):
        # This should return the content of the buffer within head-n_last (inclusive) ... head (exclusive)
        # How to vectorize this?
        pass


class AtariEnv(Env, Serializable):
    def __init__(
            self,
            game,
            obs_type="image",
            img_width=84,
            img_height=84,
            frame_skip=4,
            n_last_screens=4,
            terminate_on_life_lost=False,
            reset_on_life_lost=False,
    ):
        """
        :param game: Name of the game, without the ".bin" file extension.
        :param obs_type: Either "image" or "ram".
        :param img_width: Width of the image observation.
        :param img_height: Height of the image observation.
        :param frame_skip: How many frames to skip, during with the same action will be applied.
        :param n_last_screens: Number of the most recent screens to be used as part of the observation.
        :param terminate_on_life_lost: For games which count the number of lives, whether to count as terminal once a
        life is lost.
        :param reset_on_life_lost: For games which count the number of lives, whether to reset once a life is lost.
        Not resetting provides two benefits: (1) faster execution and (2) exposure to a potentially wider state
        distribution. This is only effective when running the environment in a vectorized form. Functions like
        rollout() will still attempt to reset the environment whenever the terminal signal is indicated.
        :return:
        """
        Serializable.quick_init(self, locals())
        self.game = game
        self.img_width = img_width
        self.img_height = img_height
        self.frame_skip = frame_skip
        self.obs_type = obs_type
        self.n_last_screens = n_last_screens
        self.terminate_on_life_lost = terminate_on_life_lost
        self.reset_on_life_lost = reset_on_life_lost
        if reset_on_life_lost:
            assert terminate_on_life_lost
        self.executor = VecAtariEnv(n_envs=1, env=self)
        self.reset()

    def reset(self):
        return self.executor.reset(dones=None)[0]

    def step(self, action):
        next_obses, rewards, dones, infos = self.executor.step([action], max_path_length=None)
        return Step(next_obses[0], rewards[0], dones[0], **{k: v[0] for k, v in infos.items()})

    @cached_property
    def observation_space(self):
        if self.obs_type == "image":
            return Box(low=-1, high=1, shape=(self.img_height, self.img_width, self.n_last_screens))
        elif self.obs_type == "ram":
            return Box(low=-1, high=1, shape=(self.executor.ram_size,))
        else:
            raise NotImplementedError

    @cached_property
    def action_space(self):
        return Discrete(len(self.executor.legal_actions))

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs):
        return VecAtariEnv(n_envs=n_envs, env=self)

    def get_image_obs(self, rescale=False, resize=False):
        return self.executor.get_image_obs(rescale=rescale, resize=resize)[0]

    def get_ram(self, rescale=False):
        return self.executor.get_ram_obs(rescale=rescale)[0]


class VecAtariEnv(object):
    def __init__(self, n_envs, env):
        self.n_envs = n_envs
        self.env = env
        self.ts = np.zeros((n_envs,), dtype=np.int)
        self.lives = np.zeros((n_envs,), dtype=np.int)
        self.par_games = None
        self.screen_width = None
        self.screen_height = None
        self.ram_size = None
        self.legal_actions = None
        self.screen_buffer = None
        self.configure()
        self.reset(return_obs=False)

    def configure(self):
        self.par_games = par_ale.ParAtari(
            n_envs=self.n_envs,
            rom_path=atari_py.get_game_path(self.env.game).encode()
        )
        for idx in range(self.n_envs):
            self.par_games.set_float(idx, b'repeat_action_probability', 0.0)
            self.par_games.set_bool(idx, b'color_averaging', False)
            self.par_games.set_bool(idx, b'sound', False)  # Sound doesn't work on OSX
            seed = np.random.randint(0, 2 ** 16)
            self.par_games.set_int(idx, b'random_seed', seed)
        self.screen_width = self.par_games.get_screen_width(0)
        self.screen_height = self.par_games.get_screen_height(0)
        self.ram_size = self.par_games.get_ram_size(0)
        self.legal_actions = np.asarray(self.par_games.get_minimal_action_set(0))
        # Implement this as a circular buffer within numpy, to speed up operations (avoiding constructions of new
        # arrays when possible)
        if self.env.obs_type == "image":
            self.screen_buffer = NaiveCircularBuffer(
                batch_size=self.n_envs,
                buffer_size=self.env.n_last_screens,
                data_shape=(self.env.img_height, self.env.img_width),
                dtype=np.float32
            )
        if self.screen_height == 250:
            raise RuntimeError("This ROM is for PAL. Please use ROMs for NTSC")

    def reset(self, dones=None, return_obs=True):
        if dones is None:
            dones = np.asarray([True] * self.n_envs)
        else:
            dones = np.cast['bool'](dones)
        if np.any(dones):
            self.par_games.reset_game_all(dones)
            lives = np.empty((self.n_envs,), dtype=np.intc)
            self.par_games.get_lives_all(lives)
            self.ts[dones] = 0
            self.lives[dones] = lives[dones]
            if self.env.obs_type == "image":
                self.reset_buffer(dones)
                self.record_frame(dones)
        if return_obs:
            return self.get_current_obs()[dones]

    def reset_buffer(self, mask):
        self.screen_buffer.reset(mask=mask)

    def record_frame(self, mask):
        data = self.get_image_obs(rescale=True, resize=True, mask=mask)
        self.screen_buffer.push(data=data, mask=mask)

    @property
    def num_envs(self):
        return self.n_envs

    def get_image_obs(self, rescale=False, resize=False, mask=None):
        if mask is None:
            mask = [True] * self.n_envs
        mask = np.cast['bool'](mask)
        output_buffer = np.empty((len(mask), self.screen_height * self.screen_width), dtype=np.uint8)
        self.par_games.get_game_screen_all(output_buffer, mask)
        output_buffer = output_buffer.reshape((len(mask), self.screen_height, self.screen_width))
        if resize:
            images = decode_gray(output_buffer)
            ret = np.empty((len(mask), self.env.img_height, self.env.img_width), dtype=np.float)
            for idx, img in enumerate(images):
                ret[idx] = cv2.resize(
                    img,
                    (self.env.img_height, self.env.img_width),
                    interpolation=cv2.INTER_LINEAR
                )
        else:
            ret = rgb_palette[output_buffer]
        if rescale:
            # ret /= 255.0
            # ret -= 0.5
            # ret *= 2.
            return ret# / 255.0 - 0.5) * 2.
        else:
            return np.cast['uint8'](ret * 255.0)

    def get_ram_obs(self, rescale=False):
        output_buffer = np.empty((self.n_envs, self.ram_size), dtype=np.uint8)
        self.par_games.get_ram_all(output_buffer)
        if rescale:
            output_buffer = (output_buffer * 1.0 / 255 - 0.5) * 2
        return output_buffer

    def get_current_obs(self):
        if self.env.obs_type == "image":
            imgs = self.screen_buffer.last(self.env.n_last_screens)
            imgs = np.transpose(imgs, [0, 2, 3, 1])
            return imgs
        elif self.env.obs_type == "ram":
            return self.get_ram_obs(rescale=True)
        else:
            raise NotImplementedError

    def step(self, action_n, max_path_length):
        rewards = np.empty((self.n_envs,), dtype=np.intc)
        actions = np.cast['intc'](self.legal_actions[np.asarray(action_n)])
        self.par_games.act_all(
            frame_skip=self.env.frame_skip,
            actions=actions,
            out_rewards=rewards
        )
        game_over = np.empty((self.n_envs,), dtype=np.uint8)
        lives = np.empty((self.n_envs,), dtype=np.intc)

        self.par_games.get_game_over_all(game_over)
        self.par_games.get_lives_all(lives)

        if self.env.terminate_on_life_lost:
            dones = np.logical_or(game_over, lives < self.lives)
        else:
            dones = game_over

        if self.env.reset_on_life_lost:
            reset_dones = np.logical_or(game_over, lives < self.lives)
        else:
            reset_dones = game_over

        self.ts += 1
        self.lives[:] = lives
        if max_path_length is not None:
            dones[self.ts >= max_path_length] = True
            reset_dones[self.ts >= max_path_length] = True

        if np.any(reset_dones):
            self.reset(reset_dones, return_obs=False)

        # for the rest, just advance the recording
        if self.env.obs_type == "image":
            self.record_frame(np.logical_not(reset_dones))

        next_obs = self.get_current_obs()

        return next_obs, rewards, dones, dict()

    def terminate(self):
        pass


if __name__ == "__main__":
    env = AtariEnv('breakout')

    import cv2

    env.step(env.action_space.sample())
    img = env.get_image_obs(rescale=False, resize=False)

    cv2.imshow('image', img)
    cv2.waitKey(0)
