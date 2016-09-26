from rllab.envs.base import Env, Step
from vizdoom import DoomGame
from vizdoom import ScreenResolution
from vizdoom import ScreenFormat
from vizdoom import Button
from vizdoom import Mode
import os
import uuid
import numpy as np
import cv2
from rllab import config
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from sandbox.rocky.neural_learner.doom_utils.wad import WAD
import atexit

DOOM_PATH = os.environ["DOOM_PATH"]

ACTIONS = [
    [True, False, False, False],
    [False, True, False, False],
    [False, False, True, False],
    [False, False, False, True],
]


class DoomGoalFindingMazeEnv(Env):
    def __init__(self):
        self.game = None
        self.reset_trial()
        atexit.register(self.terminate)

    def terminate(self):
        atexit.unregister(self.terminate)
        if self.game is not None:
            self.game.close()
        self.game = None

    def reset_trial(self):
        if self.game is not None:
            self.game.close()
        self.game = self.init_game()
        return self.reset()

    def init_game(self):
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

    def reset(self):
        self.game.new_episode()
        return self.get_image_obs(rescaled=True)

    def step(self, action):
        reward = self.game.make_action(ACTIONS[action])
        done = self.game.is_episode_finished()
        next_obs = self.get_image_obs(rescaled=True)
        return Step(next_obs, reward, done)

    def get_image_obs(self, rescaled=False):
        image = np.copy(self.game.get_state().image_buffer)
        if rescaled:
            image = (image / 255.0 - 0.5) * 2.0
        return image

    def render(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', width=400, height=400)
        cv2.imshow('image', cv2.resize(self.get_image_obs(rescaled=False), (400, 400)))
        cv2.waitKey(10)

    @property
    def observation_space(self):
        return Box(low=-1., high=1., shape=(120, 160, 3))

    @property
    def action_space(self):
        return Discrete(4)
