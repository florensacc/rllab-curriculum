from io import StringIO

import vizdoom
from cached_property import cached_property
import os
import cv2
import uuid
import numpy as np
from rllab import config
from rllab.core.serializable import Serializable
from vizdoom import ScreenResolution
from vizdoom import ScreenFormat
from vizdoom import Button
from vizdoom import Mode
from vizdoom import GameVariable

from rllab.misc import logger
from sandbox.rocky.neural_learner.doom_utils.wad import WAD
from sandbox.rocky.neural_learner.envs.doom_env import DoomEnv
from sandbox.rocky.neural_learner.envs.doom_hex_utils import mkwad
from sandbox.rocky.s3.resource_manager import resource_manager
import io
from sandbox.rocky.neural_learner.doom_utils.wad import WAD, Lump, compile_script
from rllab.misc.ext import using_seed


class DoomHexGoalFindingMazeEnv(DoomEnv, Serializable):
    def __init__(
            self,
            map_seed=0,
            n_trajs=1000,
            version="v7",
            side_length=120,
            margin=60,
            offset_x=1000,
            offset_y=1000,
            tolerance=20,
            n_repeats=2,
            difficulty=1,
            alive_reward=1.0,
            randomize_texture=True,
            living_reward=-0.01,
            n_targets=1,
            scale=1,
            allow_backwards=True,
            *args,
            **kwargs):
        Serializable.quick_init(self, locals())
        self.living_reward = living_reward
        self.allow_backwards = allow_backwards
        self.doom_scenario_path = resource_manager.get_file(
            *mkwad(
                seed=map_seed, n_trajs=n_trajs, version=version,
                side_length=side_length, margin=margin, offset_x=offset_x, offset_y=offset_y, tolerance=tolerance,
                n_repeats=n_repeats, difficulty=difficulty, alive_reward=alive_reward,
                randomize_texture=randomize_texture, n_targets=n_targets, scale=scale
            ),
            compress=True
        )
        print(self.doom_scenario_path)
        wad = WAD.from_file(self.doom_scenario_path)
        self.wad = wad
        self.level_names = [l.name.encode() for l in wad.levels]
        super().__init__(
            *args,
            **dict(kwargs, restart_game=False, restart_game_on_reset_trial=False)
        )

    def get_doom_config(self):
        DOOM_PATH = os.environ["DOOM_PATH"]
        doom_config = DoomEnv.get_doom_config(self)
        doom_config.vizdoom_path = os.path.join(DOOM_PATH, "bin/vizdoom").encode()
        doom_config.doom_game_path = os.path.join(DOOM_PATH, "scenarios/freedoom2.wad").encode()
        doom_config.doom_scenario_path = self.doom_scenario_path.encode()
        doom_config.doom_map = np.random.choice(self.level_names)
        doom_config.render_hud = False
        doom_config.render_crosshair = False
        doom_config.render_weapon = False
        doom_config.render_decals = False
        doom_config.render_particles = False
        doom_config.living_reward = self.living_reward
        if self.allow_backwards:
            doom_config.available_buttons = [
                Button.TURN_LEFT,
                Button.TURN_RIGHT,
                Button.MOVE_FORWARD,
                Button.MOVE_BACKWARD
            ]
        else:
            doom_config.available_buttons = [
                Button.TURN_LEFT,
                Button.TURN_RIGHT,
                Button.MOVE_FORWARD,
            ]
        return doom_config

    @cached_property
    def action_map(self):
        if self.allow_backwards:
            return np.asarray([
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ], dtype=np.intc)
        else:
            return np.asarray([
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ], dtype=np.intc)

    def log_diagnostics(self, paths):
        success_rate = np.mean([path["rewards"][-1] > 0 for path in paths])
        success_traj_len = np.asarray([len(path["rewards"]) for path in paths if path["rewards"][-1] > 0])
        logger.record_tabular('SuccessRate', success_rate)
        logger.record_tabular_misc_stat('SuccessTrajLen', success_traj_len, placement="front")

    def log_diagnostics_multi(self, multi_env, paths, frame_skip_specific=False):
        episode_success = [[] for _ in range(multi_env.n_episodes)]
        for path in paths:
            rewards = path['rewards']
            splitter = np.where(path['env_infos']['episode_done'])[0][:-1] + 1
            split_rewards = np.split(rewards, splitter)
            split_success = [x[-1] > 0 for x in np.split(rewards, splitter)]
            for epi, (rews, success) in enumerate(zip(split_rewards, split_success)):
                episode_success[epi].append(success)

        def log_stat(name, data):
            avg_data = list(map(np.mean, data))
            logger.record_tabular('Average%s(First)' % name, avg_data[0])
            logger.record_tabular('Average%s(Last)' % name, avg_data[-1])
            logger.record_tabular('Delta%s(Last-First)' % name, avg_data[-1] - avg_data[0])

        log_stat('SuccessRate', episode_success)

        # Log results for each frame skip
        if self.stochastic_frame_skips is not None and not frame_skip_specific:
            for frame_skip in self.stochastic_frame_skips:
                fs_paths = [p for p in paths if p["env_infos"]["frame_skip"][0] == frame_skip]
                with logger.tabular_prefix("FrameSkip[%d]|" % frame_skip):
                    self.log_diagnostics_multi(multi_env, fs_paths, frame_skip_specific=True)


    def render(self, close=False, wait_key=True):
        level_name = self.executor.par_games.get_doom_map(0).decode()
        level = [x for x in self.wad.levels if x.name == level_name][0]
        level_content = [x for x in level.lumps if x.name == "TEXTMAP"][0].content
        agent_x = vizdoom.doom_fixed_to_double(self.executor.par_games.get_game_variable(0, GameVariable.USER1))
        agent_y = vizdoom.doom_fixed_to_double(self.executor.par_games.get_game_variable(0, GameVariable.USER2))
        obs_img = self.get_image_obs(rescale=True)
        obs_img = np.cast['uint8']((obs_img + 1) * 0.5 * 255)
        obs_img = obs_img.reshape((self.observation_space.shape))
        height, width, _ = obs_img.shape
        map_img = plot_textmap(level_content, agent_x, agent_y, out_width=400, out_height=400)

        obs_img = cv2.resize(obs_img, (400, 400))
        joint_img = np.concatenate([obs_img, map_img], axis=1)
        cv2.imshow("Map", joint_img)
        if wait_key:
            cv2.waitKey(10)

    def start_interactive(self):
        while True:
            self.render(wait_key=False)
            key = int(cv2.waitKey(10))

            up = 63232
            down = 63233
            left = 63234
            right = 63235

            if key == left:
                self.step(0)
            elif key == right:
                self.step(1)
            elif key == up:
                self.step(2)
            elif key == down and self.allow_backwards:
                self.step(3)
            else:
                pass


def plot_textmap(content, agent_x, agent_y, out_width, out_height):
    if isinstance(content, bytes):
        content = content.decode()
    parts = content.replace('\n', '').replace(' ', '').split('}')

    parsed_parts = []
    for part in parts:
        if len(part) > 0:
            part = part.split('{')
            type_part = part[0].split(';')[-1]
            attrs = dict([x.split('=') for x in part[1].split(';') if len(x) > 0])
        parsed_parts.append(dict(attrs, klass=type_part))

    vertices = []

    for part in parsed_parts:
        if part['klass'] == 'vertex':
            vertices.append((int(part['x']), int(part['y'])))

    vertices = np.asarray(vertices)

    min_x, min_y = np.min(vertices, axis=0)  # - 50
    max_x, max_y = np.max(vertices, axis=0)  # + 50

    height = max_x - min_x
    width = max_y - min_y
    # import ipdb; ipdb.set_trace()
    # Now plot the map

    import cv2
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    def rescale_point(x, y):
        tx = int((int(x) - min_x) / (max_x - min_x) * height)
        ty = int((int(y) - min_y) / (max_y - min_y) * width)
        return (ty, tx)  # tx, ty)

    # cx = width / 2
    # cy = height / 2

    # cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Map", width=1000, height=1000)
    vertices = []
    # n_lines = 0
    # inc = 20  # 5#3#2
    for idx, part in enumerate(parsed_parts):
        if part['klass'] == 'vertex':
            cv2.circle(img, center=rescale_point(part['x'], part['y']), color=(100, 100, 100), radius=5,
                       thickness=-1)
            vertices.append(part)
        elif part['klass'] == 'linedef':
            pt1 = rescale_point(vertices[int(part['v1'])]['x'], vertices[int(part['v1'])]['y'])
            pt2 = rescale_point(vertices[int(part['v2'])]['x'], vertices[int(part['v2'])]['y'])
            cv2.line(img, pt1, pt2, color=(0, 0, 0), thickness=5)
        elif part['klass'] == 'thing':
            if int(part['type']) == 1:
                # character
                cv2.circle(img, center=rescale_point(agent_x, agent_y), color=(255, 0, 0), radius=5,
                           thickness=-1)
            elif int(part['type']) == 5:
                cv2.circle(img, center=rescale_point(part['x'], part['y']), color=(0, 0, 255), radius=5,
                           thickness=-1)
                # blue keycard

    img = cv2.resize(img, (out_width, out_height))
    return img

    # cv2.imshow("Map", img)
    # cv2.waitKey(15)
