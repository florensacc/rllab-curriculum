from rllab.envs.base import Env, Step
import os
import uuid
import numpy as np
import cv2
from rllab import config
from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
import atexit
import pyximport

DOOM_PATH = os.environ["DOOM_PATH"]

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension


def new_get_distutils_extension(modname, pyxfilename, language_level=None):
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

ACTIONS = np.asarray([
    [True, False, False, False],
    [False, True, False, False],
    [False, False, True, False],
    [False, False, False, True],
], dtype=np.intc)


class DoomGoalFindingMazeEnv(Env, Serializable):
    def __init__(self, restart_game=True):
        Serializable.quick_init(self, locals())
        self.game = None
        self.reward_so_far = None
        self.restart_game = restart_game
        self.reset_trial()
        atexit.register(self.terminate)

    def terminate(self):
        atexit.unregister(self.terminate)
        if self.game is not None:
            self.game.close()
        self.game = None

    def reset_trial(self):
        return self.reset(restart_game=True)

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs, max_path_length):
        return VecDoomGoalFindingMazeEnv(n_envs=n_envs, max_path_length=max_path_length, restart_game=self.restart_game)

    @staticmethod
    def configure_game(game):
        # configuring the game without initializing it
        from vizdoom import DoomGame
        from vizdoom import ScreenResolution
        from vizdoom import ScreenFormat
        from vizdoom import Button
        from vizdoom import Mode
        from vizdoom import GameVariable
        from sandbox.rocky.neural_learner.doom_utils.wad import WAD

        wad = WAD.from_folder(
            os.path.join(config.PROJECT_PATH, "sandbox/rocky/neural_learner/envs/wads/goal_finding_maze"))
        wad_file_name = "/tmp/%s.wad" % uuid.uuid4()
        wad.save(wad_file_name, force=True)
        game = DoomGame()
        game.set_vizdoom_path(os.path.join(DOOM_PATH, "bin/vizdoom"))
        game.set_doom_game_path(os.path.join(DOOM_PATH, "scenarios/freedoom2.wad"))
        game.set_doom_scenario_path(wad_file_name)
        game.set_doom_map("MAP01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(False)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.set_living_reward(-0.01)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.add_available_button(Button.TURN_LEFT)
        game.add_available_button(Button.TURN_RIGHT)
        game.add_available_button(Button.MOVE_FORWARD)
        game.add_available_button(Button.MOVE_BACKWARD)
        game.set_mode(Mode.PLAYER)

    def init_game(self):
        from vizdoom import DoomGame
        from vizdoom import ScreenResolution
        from vizdoom import ScreenFormat
        from vizdoom import Button
        from vizdoom import Mode
        from vizdoom import GameVariable
        from sandbox.rocky.neural_learner.doom_utils.wad import WAD
        DOOM_PATH = os.environ["DOOM_PATH"]

        wad = WAD.from_folder(
            os.path.join(config.PROJECT_PATH, "sandbox/rocky/neural_learner/envs/wads/goal_finding_maze"))
        wad_file_name = "/tmp/%s.wad" % uuid.uuid4()
        wad.save(wad_file_name, force=True)
        game = DoomGame()
        game.set_vizdoom_path(os.path.join(DOOM_PATH, "bin/vizdoom"))
        game.set_doom_game_path(os.path.join(DOOM_PATH, "scenarios/freedoom2.wad"))
        game.set_doom_scenario_path(wad_file_name)
        game.set_doom_map("MAP01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(False)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.set_living_reward(-0.01)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.add_available_button(Button.TURN_LEFT)
        game.add_available_button(Button.TURN_RIGHT)
        game.add_available_button(Button.MOVE_FORWARD)
        game.add_available_button(Button.MOVE_BACKWARD)
        game.set_mode(Mode.PLAYER)
        game.init()
        return game

    def reset(self, restart_game=None):
        if restart_game is None:
            restart_game = self.restart_game
        if restart_game or self.game is None:
            if self.game is not None:
                self.game.close()
                self.game = None
            self.game = self.init_game()
        self.game.new_episode()
        self.reward_so_far = 0
        return self.get_image_obs(rescale=True)

    def step(self, action):
        self.game.set_action(ACTIONS[action])
        self.game.advance_action(4, True, True)
        total_reward = self.game.get_total_reward()
        delta_reward = total_reward - self.reward_so_far
        self.reward_so_far = total_reward
        done = self.game.is_episode_finished()
        next_obs = self.get_image_obs(rescale=True)
        return Step(next_obs, delta_reward, done)

    def get_image_obs(self, rescale=False):
        image = np.copy(self.game.get_game_screen())
        if rescale:
            image = (image / 255.0 - 0.5) * 2.0
        return image

    def render(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', width=400, height=400)
        cv2.imshow('image', cv2.resize(self.get_image_obs(rescale=False), (400, 400)))
        cv2.waitKey(10)

    @property
    def observation_space(self):
        return Box(low=-1., high=1., shape=(120, 160, 3))

    @property
    def action_space(self):
        return Discrete(4)


class VecDoomGoalFindingMazeEnv(object):
    def __init__(self, n_envs, max_path_length, restart_game):
        self.n_envs = n_envs
        self.restart_game = restart_game
        self.max_path_length = max_path_length
        self.par_games = par_doom.ParDoom(n_envs)
        self.rewards_so_far = np.zeros((self.n_envs,))
        self.ts = np.zeros((n_envs,), dtype=np.int)
        self.reset_trial()

    @property
    def num_envs(self):
        return self.n_envs

    def reset_trial(self):
        return self.reset(restart_game=True)

    def reset(self, dones=None, restart_game=None, return_obs=True):
        if restart_game is None:
            restart_game = self.restart_game
        if dones is None or np.any(dones):
            if restart_game:
                try:
                    self.par_games.close_all(dones)
                except Exception as e:
                    import ipdb;
                    ipdb.set_trace()
            self.init_games(dones)
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
        if rescale:
            images = np.array(images, dtype=np.float32)
            images /= 255.0  # [0, 1]
            images -= 0.5  # [-0.5, 0.5]
            images *= 2.0  # [-1, 1]
        else:
            images = np.array(images)
        return images

    def step(self, action_n):
        for i in range(self.n_envs):
            self.par_games.set_action(i, ACTIONS[action_n[i]])
        # advance in parallel
        self.par_games.advance_action_all(4, True, True)
        dones = np.asarray(
            [self.par_games.is_episode_finished(i) for i in range(self.n_envs)],
            dtype=np.uint8
        )
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = 1

        next_obs = self.get_image_obs(rescale=True)

        total_rewards = np.asarray(
            [self.par_games.get_total_reward(i) for i in range(self.n_envs)]
        )

        delta_rewards = total_rewards - self.rewards_so_far

        self.rewards_so_far = total_rewards

        if np.any(dones):
            self.reset(dones)

        return next_obs, delta_rewards, dones, dict()

    def init_games(self, dones=None):
        from vizdoom import DoomGame
        from vizdoom import ScreenResolution
        from vizdoom import ScreenFormat
        from vizdoom import Button
        from vizdoom import Mode
        from vizdoom import GameVariable
        from sandbox.rocky.neural_learner.doom_utils.wad import WAD
        DOOM_PATH = os.environ["DOOM_PATH"]

        wad = WAD.from_folder(
            os.path.join(config.PROJECT_PATH, "sandbox/rocky/neural_learner/envs/wads/goal_finding_maze"))
        wad_file_name = "/tmp/%s.wad" % uuid.uuid4()
        wad.save(wad_file_name, force=True)

        self.par_games.create_all(dones)

        for i in range(self.n_envs):
            if dones is None or dones[i]:
                self.par_games.set_vizdoom_path(i, os.path.join(DOOM_PATH, "bin/vizdoom").encode("utf-8"))
                self.par_games.set_doom_game_path(i, os.path.join(DOOM_PATH, "scenarios/freedoom2.wad").encode("utf-8"))
                self.par_games.set_doom_scenario_path(i, wad_file_name.encode("utf-8"))
                self.par_games.set_doom_map(i, b"MAP01")
                self.par_games.set_screen_resolution(i, ScreenResolution.RES_160X120)
                self.par_games.set_screen_format(i, ScreenFormat.RGB24)
                self.par_games.set_render_hud(i, False)
                self.par_games.set_render_crosshair(i, False)
                self.par_games.set_render_weapon(i, False)
                self.par_games.set_render_decals(i, False)
                self.par_games.set_render_particles(i, False)
                self.par_games.set_living_reward(i, -0.01)
                self.par_games.set_window_visible(i, False)
                self.par_games.set_sound_enabled(i, False)
                self.par_games.add_available_button(i, Button.TURN_LEFT)
                self.par_games.add_available_button(i, Button.TURN_RIGHT)
                self.par_games.add_available_button(i, Button.MOVE_FORWARD)
                self.par_games.add_available_button(i, Button.MOVE_BACKWARD)
                self.par_games.set_mode(i, Mode.PLAYER)

        self.par_games.init_all(dones)
