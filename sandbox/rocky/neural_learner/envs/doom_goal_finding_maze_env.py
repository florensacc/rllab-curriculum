from io import StringIO

import cv2
import vizdoom
from cached_property import cached_property
import os
import uuid
import numpy as np
from rllab import config
from rllab.core.serializable import Serializable
from vizdoom import ScreenResolution
from vizdoom import ScreenFormat
from vizdoom import Button
from vizdoom import Mode

from rllab.misc import logger
from sandbox.rocky.neural_learner.doom_utils.wad import WAD
from sandbox.rocky.neural_learner.envs.doom_env import DoomEnv
from sandbox.rocky.s3.resource_manager import resource_manager
import io
from sandbox.rocky.neural_learner.doom_utils.wad import WAD, Lump, compile_script
from rllab.misc.ext import using_seed

SCRIPT = """
#include "zcommon.acs"

int target_id = 2;
global int 0:reward;
global int 1:x_pos;
global int 2:y_pos;


script 1 OPEN
{
    reward = 0;
    SetThingSpecial(target_id, ACS_ExecuteAlways, 2);
}


script 2 (void)
{
    reward = reward + 100.0;
    Exit_Normal(0);
}

script 3 ENTER
{
    while(1)
    {
        delay(1);
        x_pos = GetActorX(0);
        y_pos = GetActorY(0);
    }
}
"""


def create_map(maze_sizes=None, *args, **kwargs):
    from sandbox.rocky.neural_learner.doom_utils.textmap import Thing, Textmap, Vertex, Linedef, Sidedef, Sector
    from sandbox.rocky.neural_learner.envs.maze.dfs_grid_maze_generator import DFSGridMazeGenerator

    player = Thing(x=0, y=0, type=1, id=1)
    # this is the goal the player should reach
    # some candidate things:
    # 5 - blue keycard (kind of too small)

    bluecard = Thing(x=0, y=0, z=20, type=5, id=2)

    textmap = Textmap(namespace="zdoom", items=[])

    things = [player, bluecard]

    maze_gen = DFSGridMazeGenerator()

    if maze_sizes is None:
        maze_sizes = [5, 7, 9]

    maze_size = np.random.choice(maze_sizes)
    maze = maze_gen.gen_maze(n_row=maze_size, n_col=maze_size)

    linedefs = []
    sidedefs = [Sidedef(sector=0, texturemiddle="BRICK9")]
    sectors = []
    vertices = []

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i][j] == 1:  # if not wall
                vs = [
                    (96 * i, 96 * (j + 1)),
                    (96 * (i + 1), 96 * (j + 1)),
                    (96 * (i + 1), 96 * j),
                    (96 * i, 96 * j)
                ]
                for v in vs:
                    if v not in vertices:
                        vertices.append(v)
                ls = []
                for vid in range(4):
                    vid_next = (vid + 1) % 4
                    ls.append((vertices.index(vs[vid]), vertices.index(vs[vid_next])))
                for l in ls:
                    if l not in linedefs:
                        linedefs.append(l)

    sectors.append(
        Sector(
            texturefloor="FLAT3",
            textureceiling="FLAT2",
            heightceiling=104,
            lightlevel=210,
        )
    )

    linedefs = [l for l in linedefs if (l[1], l[0]) not in linedefs]

    textmap.items.extend(things)
    textmap.items.extend([Vertex(x=v[0], y=v[1]) for v in vertices])
    textmap.items.extend([Linedef(v1=l[0], v2=l[1], blocking=True, sidefront=0) for l in linedefs])
    textmap.items.extend(sidedefs)
    textmap.items.extend(sectors)

    for thing in things:
        while True:
            x = np.random.randint(0, maze.shape[0] * 96)
            y = np.random.randint(0, maze.shape[1] * 96)
            i = x // 96
            j = y // 96
            # make sure it's not too close to the boundary
            if x % 96 > 10 and x % 96 < 86 and y % 96 > 10 and y % 96 < 86:
                if maze[i][j] == 1:
                    thing.x = x
                    thing.y = y
                    break

    sio = StringIO()
    textmap.write(sio)
    return sio.getvalue()


def mkwad(seed, n_trajs, version):
    resource_name = "doom/goal_finding_maze/n{n_trajs}_s{seed}_{version}.wad".format(
        n_trajs=n_trajs, seed=seed, version=version)

    def mk():
        with using_seed(seed):
            script = SCRIPT

            behavior = compile_script(script.encode())

            wad = WAD()
            wad.wad_type = "PWAD"

            level_names = ["MAP%02d" % idx for idx in range(n_trajs)]

            for level_name in level_names:
                wad.lumps.append(Lump(name=level_name, content=b''))
                wad.lumps.append(Lump(name="TEXTMAP", content=create_map().encode()))
                wad.lumps.append(Lump(name="SCRIPT", content=script.encode()))
                wad.lumps.append(Lump(name="BEHAVIOR", content=behavior))
                wad.lumps.append(Lump(name="ENDMAP", content=b''))

            wad.reorganize()

            bio = io.BytesIO()

            wad.save_io(bio)

            bytes = bio.getvalue()

            print("uploading...")
            resource_manager.register_data(resource_name, bytes)
            print("uploaded")

    return resource_name, mk


class DoomGoalFindingMazeEnv(DoomEnv, Serializable):
    def __init__(self, living_reward=-0.01, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.living_reward = living_reward
        self.doom_scenario_path = resource_manager.get_file(*mkwad(seed=0, n_trajs=1000, version="v2"))
        # annotation_path = resource_manager.get_file(*mkwad_annotation(seed=0, n_trajs=1000, version="v1"))
        wad = WAD.from_file(self.doom_scenario_path)
        self.wad = wad
        self.level_names = [l.name.encode() for l in wad.levels]
        super().__init__(*args, **dict(kwargs, restart_game=False, restart_game_on_reset_trial=False))

    def get_doom_config(self):
        DOOM_PATH = os.environ["DOOM_PATH"]
        doom_config = DoomEnv.get_doom_config(self)
        doom_config.vizdoom_path = os.path.join(DOOM_PATH, "bin/vizdoom").encode()
        doom_config.doom_game_path = os.path.join(DOOM_PATH, "scenarios/freedoom2.wad").encode()
        doom_config.doom_scenario_path = self.doom_scenario_path.encode()
        level_name = np.random.choice(self.level_names)
        doom_config.doom_map = level_name
        doom_config.render_hud = False
        doom_config.render_crosshair = False
        doom_config.render_weapon = False
        doom_config.render_decals = False
        doom_config.render_particles = False
        doom_config.living_reward = self.living_reward
        doom_config.available_buttons = [
            Button.TURN_LEFT,
            Button.TURN_RIGHT,
            Button.MOVE_FORWARD,
            Button.MOVE_BACKWARD
        ]
        return doom_config

    @cached_property
    def action_map(self):
        return np.asarray([
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ], dtype=np.intc)

    def log_diagnostics(self, paths):
        success_rate = np.mean([path["rewards"][-1] > 0 for path in paths])
        success_traj_len = np.asarray([len(path["rewards"]) for path in paths if path["rewards"][-1] > 0])
        logger.record_tabular('SuccessRate', success_rate)
        logger.record_tabular_misc_stat('SuccessTrajLen', success_traj_len)   #

    def log_diagnostics_multi(self, multi_env, paths):
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


    def render(self, close=False):
        level = [x for x in self.wad.levels if x.name == self.level_name][0]
        level_content = [x for x in level.lumps if x.name == "TEXTMAP"][0].content
        agent_x = vizdoom.doom_fixed_to_double(self.executor.par_games.get_game_variable(0, vizdoom.GameVariable.USER1))
        agent_y = vizdoom.doom_fixed_to_double(self.executor.par_games.get_game_variable(0, vizdoom.GameVariable.USER2))
        obs_img = self.get_image_obs(rescale=True)
        obs_img = np.cast['uint8']((obs_img + 1) * 0.5 * 255)
        obs_img = obs_img.reshape((self.observation_space.shape))
        height, width, _ = obs_img.shape
        map_img = plot_textmap(level_content, agent_x, agent_y, out_width=400, out_height=400)

        obs_img = cv2.resize(obs_img, (400, 400))
        joint_img = np.concatenate([obs_img, map_img], axis=1)
        cv2.imshow("Map", joint_img)
        cv2.waitKey(10)


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

    min_x, min_y = np.min(vertices, axis=0) - 50
    max_x, max_y = np.max(vertices, axis=0) + 50


    height = max_x - min_x
    width = max_y - min_y
    # import ipdb; ipdb.set_trace()
    # Now plot the map

    import cv2
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    def rescale_point(x, y):
        tx = int((int(x) - min_x) / (max_x - min_x) * height)
        ty = int((int(y) - min_y) / (max_y - min_y) * width)
        return (ty, tx)#tx, ty)

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

