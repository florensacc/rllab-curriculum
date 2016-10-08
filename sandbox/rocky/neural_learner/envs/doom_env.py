from rllab.envs.base import Env, Step
import os
import numpy as np
import cv2

from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
import atexit
import pyximport
from vizdoom import Mode, ScreenResolution, ScreenFormat, Button

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension


def new_get_distutils_extension(modname, pyxfilename, language_level=None):
    DOOM_PATH = os.environ["DOOM_PATH"]
    extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language = 'c++'
    extension_mod.library_dirs = [os.path.join(DOOM_PATH, "bin")]
    extension_mod.runtime_library_dirs = [os.path.join(DOOM_PATH, "bin")]
    extension_mod.include_dirs = [os.path.join(DOOM_PATH, "include")]
    extension_mod.extra_compile_args = ['-fopenmp']
    extension_mod.extra_link_args = ['-fopenmp']
    extension_mod.libraries = ["vizdoom"]

    return extension_mod, dict(setup_args, script_args=['--verbose'])  # verbose=True)


pyximport.pyximport.get_distutils_extension = new_get_distutils_extension
pyximport.install()

from sandbox.rocky.neural_learner.doom_utils import par_doom


class DoomConfig(object):
    def __init__(self):
        self.vizdoom_path = None
        self.doom_game_path = None
        self.doom_scenario_path = None
        self.screen_resolution = None
        self.screen_format = None
        self.render_hud = None
        self.render_crosshair = None
        self.render_weapon = None
        self.render_decals = None
        self.render_particles = None
        self.living_reward = None
        self.window_visible = None
        self.sound_enabled = None
        self.available_buttons = None
        self.mode = None

    def configure(self, par_games, i):
        if self.vizdoom_path is not None:
            par_games.set_vizdoom_path(i, self.vizdoom_path)
        if self.doom_game_path is not None:
            par_games.set_doom_game_path(i, self.doom_game_path)
        if self.doom_scenario_path is not None:
            par_games.set_doom_scenario_path(i, self.doom_scenario_path)
        if self.screen_resolution is not None:
            par_games.set_screen_resolution(i, self.screen_resolution)
        if self.screen_format is not None:
            par_games.set_screen_format(i, self.screen_format)
        if self.render_hud is not None:
            par_games.set_render_hud(i, self.render_hud)
        if self.render_crosshair is not None:
            par_games.set_render_crosshair(i, self.render_crosshair)
        if self.render_weapon is not None:
            par_games.set_render_weapon(i, self.render_weapon)
        if self.render_decals is not None:
            par_games.set_render_decals(i, self.render_decals)
        if self.render_particles is not None:
            par_games.set_render_particles(i, self.render_particles)
        if self.living_reward is not None:
            par_games.set_living_reward(i, self.living_reward)
        if self.window_visible is not None:
            par_games.set_window_visible(i, self.window_visible)
        if self.sound_enabled is not None:
            par_games.set_sound_enabled(i, self.sound_enabled)
        # generate new seed
        seed = np.random.randint(low=0, high=np.iinfo(np.uintc).max)
        par_games.set_seed(i, seed)
        if self.available_buttons is not None:
            for button in self.available_buttons:
                par_games.add_available_button(i, button)
        if self.mode is not None:
            par_games.set_mode(i, self.mode)


class DoomEnv(Env, Serializable):
    def __init__(self, restart_game=True, vectorized=True, verbose_debug=False, rescale_obs=None):
        Serializable.quick_init(self, locals())
        self._vectorized = vectorized
        self._verbose_debug = verbose_debug
        self.mode = Mode.PLAYER
        self.restart_game = restart_game
        self.rescale_obs = rescale_obs
        self.executor = VecDoomEnv(n_envs=1, env=self)
        self.reset_trial()

    def reset_trial(self):
        return self.reset(restart_game=True)

    @property
    def vectorized(self):
        return self._vectorized

    def vec_env_executor(self, n_envs):
        return VecDoomEnv(n_envs=n_envs, env=self)

    def get_doom_config(self):
        doom_config = DoomConfig()

        if self.mode in [Mode.ASYNC_SPECTATOR, Mode.SPECTATOR]:
            doom_config.screen_resolution = ScreenResolution.RES_1024X768
        else:
            doom_config.screen_resolution = ScreenResolution.RES_160X120

        doom_config.screen_format = ScreenFormat.RGB24
        doom_config.render_hud = True
        doom_config.render_crosshair = True
        doom_config.render_weapon = True
        doom_config.render_decals = True
        doom_config.render_particles = True
        doom_config.living_reward = 0.
        doom_config.window_visible = self.mode in [Mode.ASYNC_SPECTATOR, Mode.SPECTATOR]
        doom_config.sound_enabled = False
        doom_config.available_buttons = [
            Button.TURN_LEFT,
            Button.TURN_RIGHT,
            Button.MOVE_FORWARD,
            Button.MOVE_BACKWARD,
            Button.ATTACK,
            Button.JUMP,
        ]
        doom_config.mode = self.mode
        return doom_config

    def reset(self, restart_game=None):
        return self.executor.reset(dones=None, restart_game=restart_game)[0]

    def step(self, action):
        if self._verbose_debug:
            logger.log("start stepping")
        next_obses, rewards, dones, infos = self.executor.step([action], max_path_length=None)
        if self._verbose_debug:
            logger.log("finished stepping")
        return Step(next_obses[0], rewards[0], dones[0], **{k: v[0] for k, v in infos.items()})

    def configure_game(self, par_games, i):
        raise NotImplementedError

    def start_interactive(self):
        from vizdoom import Mode
        self.mode = Mode.SPECTATOR
        self.executor.init_games()
        while True:
            if self.executor.par_games.is_episode_finished(0):
                self.executor.par_games.close_all()
                self.executor.init_games()
                self.executor.par_games.new_episode_all()
            self.executor.par_games.advance_action_all(1, update_state=True, render_only=True)
            import time
            time.sleep(0.028)

    def get_image_obs(self, rescale=False):
        return self.executor.get_image_obs(rescale=rescale)[0]

    def render(self, close=False):
        if close:
            cv2.destroyWindow('image')
        else:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', width=400, height=400)
            cv2.imshow('image', cv2.resize(self.get_image_obs(rescale=False), (400, 400)))
            cv2.waitKey(10)

    @property
    def observation_space(self):
        if self.rescale_obs is not None:
            return Box(low=-1., high=1., shape=self.rescale_obs + (3,))
        else:
            return Box(low=-1., high=1., shape=(120, 160, 3))

    @property
    def action_map(self):
        raise NotImplementedError

    @property
    def action_space(self):
        return Discrete(len(self.action_map))


class VecDoomEnv(object):
    def __init__(
            self,
            n_envs,
            env):
        self.n_envs = n_envs
        self.env = env
        self.par_games = par_doom.ParDoom(n_envs)
        self.rewards_so_far = np.zeros((self.n_envs,))
        self.ts = np.zeros((n_envs,), dtype=np.int)
        self.reset_trial()
        atexit.register(self.terminate)

    def terminate(self):
        atexit.unregister(self.terminate)
        self.par_games.close_all()

    @property
    def num_envs(self):
        return self.n_envs

    def reset_trial(self):
        return self.reset(restart_game=True)

    def reset(self, dones=None, restart_game=None, return_obs=True):
        if restart_game is None:
            restart_game = self.env.restart_game
        if dones is None or np.any(dones):
            if restart_game:
                if self.env._verbose_debug:
                    logger.log("closing games")
                try:
                    self.par_games.close_all(dones)
                except Exception as e:
                    print(e)
                    import ipdb;
                    ipdb.set_trace()
            self.init_games(dones)
            if self.env._verbose_debug:
                logger.log("start new episodes")
            self.par_games.new_episode_all(dones)
            if dones is None:
                self.rewards_so_far[:] = 0
                self.ts[:] = 0
            else:
                self.rewards_so_far[np.cast['bool'](dones)] = 0
                self.ts[np.cast['bool'](dones)] = 0
        if return_obs:
            return self.get_image_obs(rescale=True)

    def get_image_obs(self, rescale=False):
        images = self.par_games.get_game_screen_all()
        if self.env.rescale_obs is not None:
            import cv2
            images = [cv2.resize(img, self.env.rescale_obs) for img in images]
        if rescale:
            images = np.array(images, dtype=np.float32)
            images /= 255.0  # [0, 1]
            images -= 0.5  # [-0.5, 0.5]
            images *= 2.0  # [-1, 1]
        else:
            images = np.array(images)
        return images

    def step(self, action_n, max_path_length):
        if self.env._verbose_debug:
            logger.log("setting action")
        for i in range(self.n_envs):
            self.par_games.set_action(i, self.env.action_map[action_n[i]])
        # advance in parallel
        if self.env._verbose_debug:
            logger.log("advancing action")
        self.par_games.advance_action_all(4, True, True)
        if self.env._verbose_debug:
            logger.log("checking if episode finished")
        dones = np.asarray(
            [self.par_games.is_episode_finished(i) for i in range(self.n_envs)],
            dtype=np.int32
        )
        self.ts += 1
        if max_path_length is not None:
            dones[self.ts >= max_path_length] = 1

        next_obs = self.get_image_obs(rescale=True)

        total_rewards = np.asarray(
            [self.par_games.get_total_reward(i) for i in range(self.n_envs)]
        )

        delta_rewards = total_rewards - self.rewards_so_far

        self.rewards_so_far = total_rewards

        if np.any(dones):
            if self.env._verbose_debug:
                logger.log("resetting")
            self.reset(dones)

        return next_obs, delta_rewards, np.cast['bool'](dones), dict()

    def init_games(self, dones=None):
        self.par_games.create_all(dones)

        for i in range(self.n_envs):
            if dones is None or dones[i]:
                doom_config = self.env.get_doom_config()
                doom_config.configure(self.par_games, i)
        if self.env._verbose_debug:
            logger.log("initing games")
        self.par_games.init_all(dones)
        if self.env._verbose_debug:
            logger.log("init finished")
