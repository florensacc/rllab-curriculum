from collections import OrderedDict
from io import StringIO
from string import Template

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
from rllab.spaces import Discrete
from sandbox.rocky.neural_learner.doom_utils.wad import WAD
from sandbox.rocky.neural_learner.envs.doom_default_wad_env import DoomDefaultWadEnv
from sandbox.rocky.neural_learner.envs.doom_env import DoomEnv, VecDoomEnv
from sandbox.rocky.s3.resource_manager import resource_manager
import io
from sandbox.rocky.neural_learner.doom_utils.wad import WAD, Lump, compile_script
from rllab.misc.ext import using_seed

SCRIPT = """
#include "zcommon.acs"

global int 0:reward;
global int 1:x_pos;
global int 2:y_pos;


script 1 OPEN
{
    reward = 0;
}

int map_height = $map_height;
int map_width = $map_width;
int cnt = 0;
int current_map[$map_height][$map_width] = $map_description;


script 2 (void)
{
    reward = reward + 1.0;
    Exit_Normal(0);
}

script 3 ENTER
{
    while(1)
    {
        if ($use_primitives) {
            int buttons = GetPlayerInput(-1, INPUT_FORWARDMOVE);
            int turn_buttons = GetPlayerInput(-1, INPUT_YAW);

            int next_x, next_y;

            cnt = cnt + 1;

            int cur_x = GetActorX(0);
            int cur_y = GetActorY(0);
            int quant_x = FixedMul(((FixedDiv(cur_x, $side_length) >> 16) << 16), $side_length) + FixedMul($side_length, 0.5);
            int quant_y = FixedMul(((FixedDiv(cur_y, $side_length) >> 16) << 16), $side_length) + FixedMul($side_length, 0.5);
            //Print(
            //s:"cur_x: ", f:cur_x, s:", ",
            //s:"cur_y: ", f:cur_y, s:", ",
            //s:"quant_x: ", f:quant_x, s:", ",
            //s:"quant_y: ", f:quant_y, s:", "
            //);

            if (turn_buttons != 0) {
                int angle = GetActorAngle(0);
                if (turn_buttons < 0) {
                    // turning left
                    if (-0.1 < angle && angle < 0.1 || angle > 0.9) {
                        SetActorAngle(0, 0.75);
                    } else if (0.15 < angle && angle < 0.35) {
                        SetActorAngle(0, 0.);
                    } else if (0.4 < angle && angle < 0.6) {
                        SetActorAngle(0, 0.25);
                    } else if (0.65 < angle && angle < 0.85) {
                        SetActorAngle(0, 0.5);
                    } else {
                        Print(s:"Unhandled,", f:angle);
                    }
                } else {
                    // turning right
                    if (-0.1 < angle && angle < 0.1 || angle > 0.9) {
                        SetActorAngle(0, 0.25);
                    } else if (0.15 < angle && angle < 0.35) {
                        SetActorAngle(0, 0.5);
                    } else if (0.4 < angle && angle < 0.6) {
                        SetActorAngle(0, 0.75);
                    } else if (0.65 < angle && angle < 0.85) {
                        SetActorAngle(0, 0.);
                    } else {
                        Print(s:"Unhandled,", f:angle);
                    }
                }
            } else {
                if (buttons == 256) {
                    if (GetActorAngle(0) == 0.) {
                        next_x = GetActorX(0) + $side_length;
                        next_y = GetActorY(0);
                    } else if (GetActorAngle(0) == 0.25) {
                        next_x = GetActorX(0);
                        next_y = GetActorY(0) + $side_length;
                    } else if (GetActorAngle(0) == 0.5) {
                        next_x = GetActorX(0) - $side_length;
                        next_y = GetActorY(0);
                    } else if (GetActorAngle(0) == 0.75) {
                        next_x = GetActorX(0);
                        next_y = GetActorY(0) - $side_length;
                    } else if (GetActorAngle(0) == 1.) {
                        next_x = GetActorX(0) + $side_length;
                        next_y = GetActorY(0);
                    }
                } else if (buttons == -256) {
                    if (GetActorAngle(0) == 0.) {
                        next_x = GetActorX(0) - $side_length;
                        next_y = GetActorY(0);
                    } else if (GetActorAngle(0) == 0.25) {
                        next_x = GetActorX(0);
                        next_y = GetActorY(0) - $side_length;
                    } else if (GetActorAngle(0) == 0.5) {
                        next_x = GetActorX(0) + $side_length;
                        next_y = GetActorY(0);
                    } else if (GetActorAngle(0) == 0.75) {
                        next_x = GetActorX(0);
                        next_y = GetActorY(0) + $side_length;
                    } else if (GetActorAngle(0) == 1.) {
                        next_x = GetActorX(0) - $side_length;
                        next_y = GetActorY(0);
                    }
                }

                if (buttons == 256 || buttons == -256) {
                    // check validity of next position
                    int int_x = FixedDiv(next_x, $side_length) >> 16;
                    int int_y = FixedDiv(next_y, $side_length) >> 16;
                    if (0 <= int_x && int_x < map_height && 0 <= int_y && int_y < map_width) {
                        if (current_map[int_x][int_y] == 0) {
                            SetActorPosition(0, next_x, next_y, GetActorZ(0), 0);
                        } else if (current_map[int_x][int_y] == 2) {
                            reward = reward + 1.0;
                            Exit_Normal(0);
                        } else {
                            // hitting walls
                            reward = reward - ($wall_penalty);
                            SetActorPosition(0, quant_x, quant_y, GetActorZ(0), 0);
                        }
                    }
                }
            }
        }

        x_pos = GetActorX(0);
        y_pos = GetActorY(0);

        delay(1);
    }
}

script 4 (void)
{
    reward = reward - ($wall_penalty);
}
"""

TMPL = Template(SCRIPT)


def create_map(
        side_length=96,
        margin=20,
        maze_sizes=None,
        rand_angle=False,
        map_preset=None,
        *args, **kwargs):
    from sandbox.rocky.neural_learner.doom_utils.textmap import Thing, Textmap, Vertex, Linedef, Sidedef, Sector
    from sandbox.rocky.neural_learner.envs.maze.dfs_grid_maze_generator import DFSGridMazeGenerator

    player = Thing(x=0, y=0, type=1, id=1)

    from sandbox.rocky.neural_learner.envs.mazelib.mazelib import Maze
    from sandbox.rocky.neural_learner.envs.mazelib.generate.prims import Prims

    textmap = Textmap(namespace="zdoom", items=[])

    things = [player]

    if map_preset is not None:
        m = map_preset
    else:
        if maze_sizes is None:
            maze_sizes = [3]

        if isinstance(maze_sizes, str):
            maze_sizes = [int(x) for x in maze_sizes.split(",")]

        size = np.random.choice(maze_sizes)

        m = Maze()
        m.generator = Prims(size, size)
        m.generate()
        m.generate_entrances(False, False)

    linedefs = []
    sidedefs = [Sidedef(sector=0, texturemiddle="BRICK9"),
                Sidedef(sector=0, texturemiddle="FLOOR1_7", texturetop="FLOOR1_7", clipmidtex=True,
                        scalex_mid=3.0, scaley_mid=0.3)]
    sectors = []
    vertices = []

    for i in range(m.grid.shape[0]):
        for j in range(m.grid.shape[1]):
            if m.grid[i][j] == 0:  # if not wall
                vs = [
                    (side_length * i, side_length * (j + 1)),
                    (side_length * (i + 1), side_length * (j + 1)),
                    (side_length * (i + 1), side_length * j),
                    (side_length * i, side_length * j)
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

    goal_x, goal_y = m.end

    vs = [
        (side_length * goal_x + margin, side_length * (goal_y + 1) - margin),
        (side_length * (goal_x + 1) - margin, side_length * (goal_y + 1) - margin),
        (side_length * (goal_x + 1) - margin, side_length * goal_y + margin),
        (side_length * goal_x + margin, side_length * goal_y + margin)
    ]
    for v in vs:
        if v not in vertices:
            vertices.append(v)
    textmap.items.extend(things)
    textmap.items.extend([Vertex(x=v[0], y=v[1]) for v in vertices])
    textmap.items.extend([Linedef(
        v1=l[0], v2=l[1], blocking=True,
        sidefront=0, special=80, arg0=4,
        firstsideonly=True, playerpush=True,
        repeatspecial=True,
    ) for l in linedefs])

    linedefs = []
    for vid in range(4):
        vid_next = (vid + 1) % 4
        linedefs.append((vertices.index(vs[vid_next]), vertices.index(vs[vid])))
    textmap.items.extend([Vertex(x=v[0], y=v[1]) for v in vertices])
    textmap.items.extend([Linedef(
        v1=l[0], v2=l[1],
        blocking=True,  # False, blockplayers=False,
        sidefront=1, sideback=0,
        special=80,
        arg0=2, firstsideonly=True, playerpush=True
    ) for l in linedefs])

    textmap.items.extend(sidedefs)
    textmap.items.extend(sectors)

    things[0].x = m.start[0] * side_length + side_length // 2
    things[0].y = m.start[1] * side_length + side_length // 2

    if rand_angle:
        things[0].angle = np.random.randint(low=0, high=256)

    sio = StringIO()
    textmap.write(sio)
    return sio.getvalue(), m


def gen_two_goal_maze(idx):
    from sandbox.rocky.neural_learner.envs.mazelib.mazelib import Maze
    m = Maze()
    m.grid = np.asarray([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    m.start = (4, 4)
    if idx == 0:
        m.end = (1, 1)
    elif idx == 1:
        m.end = (1, 7)
    else:
        raise NotImplementedError
    return m


def gen_two_goal_maze_lv2(idx):
    from sandbox.rocky.neural_learner.envs.mazelib.mazelib import Maze
    m = Maze()
    m.grid = np.asarray([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])
    m.start = (4, 3)
    if idx == 0:
        m.end = (4, 1)
    elif idx == 1:
        m.end = (4, 5)
    else:
        raise NotImplementedError
    return m


def mkwad(
        seed=None,
        n_trajs=1000,
        version="v1",
        side_length=96,
        margin=20,
        wall_penalty=0.,
        rand_angle=False,
        maze_sizes=None,
        map_preset=None,
        use_primitives=False,
):
    kwargs = OrderedDict([
        ("n_trajs", n_trajs),
        ("side_length", side_length),
        ("margin", margin),
        ("seed", seed),
        ("maze_sizes", maze_sizes),
        ("rand_angle", rand_angle),
        ("map_preset", map_preset),
        ("wall_penalty", wall_penalty),
        ("use_primitives", use_primitives),
        ("version", version),
    ])

    if map_preset is not None:
        if map_preset.startswith("two_goal"):
            assert n_trajs == 2

    wad_resource_name = "_".join("{0}_{1}".format(k, v) for k, v in kwargs.items())

    resource_name = "doom/doom_po/{0}.wad".format(wad_resource_name)

    def mk():
        with using_seed(seed):

            wad = WAD()
            wad.wad_type = "PWAD"

            level_names = ["MAP%02d" % idx for idx in range(n_trajs)]

            if not use_primitives:
                script = TMPL.substitute(
                    wall_penalty="%.6f" % wall_penalty,
                    map_height=1,
                    map_width=1,
                    map_description="{{1}}",
                    use_primitives=0,
                    side_length="%d." % int(side_length),
                )
                behavior = compile_script(script.encode())
            else:
                script = None
                behavior = None

            for idx, level_name in enumerate(level_names):
                wad.lumps.append(Lump(name=level_name, content=b''))
                map_args = kwargs
                if map_preset is not None:
                    if map_preset == "two_goal":
                        map_args["map_preset"] = gen_two_goal_maze(idx)
                    elif map_preset == "two_goal_lv2":
                        map_args["map_preset"] = gen_two_goal_maze_lv2(idx)
                    else:
                        raise NotImplementedError
                map_content, maze = create_map(**map_args)
                wad.lumps.append(Lump(name="TEXTMAP", content=map_content.encode()))
                if use_primitives:
                    grid = np.copy(maze.grid)
                    grid[maze.end[0]][maze.end[1]] = 2
                    map_description = "{" + ",".join(
                        ["{" + ",".join(map(str, x)) + "}" for x in grid]
                    ) + "}"
                    script = TMPL.substitute(
                        wall_penalty="%.6f" % wall_penalty,
                        map_height=str(maze.grid.shape[0]),
                        map_width=str(maze.grid.shape[1]),
                        map_description=map_description,
                        use_primitives=1,
                        side_length="%d." % int(side_length),
                    )
                    behavior = compile_script(script.encode())

                wad.lumps.append(Lump(name="SCRIPT", content=script.encode()))
                wad.lumps.append(Lump(name="BEHAVIOR", content=behavior))
                wad.lumps.append(Lump(name="ENDMAP", content=b''))

            wad.reorganize()

            bio = io.BytesIO()

            wad.save_io(bio)

            bytes = bio.getvalue()

            print("uploading...")
            resource_manager.register_data(resource_name, bytes, compress=True)
            print("uploaded")

    return resource_name, mk


class DoomFixedGoalFindingMazeEnv(DoomEnv, Serializable):
    def __init__(
            self,
            living_reward=-0.01,
            margin=20,
            side_length=96,
            n_trajs=1000,
            seed=0,
            allow_backwards=True,
            version="v1",
            maze_sizes=None,
            rand_angle=False,
            map_preset=None,
            wall_penalty=0.,
            use_primitives=False,
            *args, **kwargs
    ):
        if use_primitives:
            assert kwargs.get("frame_skip", 4) == 1
        Serializable.quick_init(self, locals())
        self.living_reward = living_reward
        self.traj = []
        self.allow_backwards = allow_backwards
        self.doom_scenario_path = resource_manager.get_file(
            *mkwad(
                seed=seed, n_trajs=n_trajs, version=version, side_length=side_length, margin=margin,
                rand_angle=rand_angle, maze_sizes=maze_sizes, map_preset=map_preset, wall_penalty=wall_penalty,
                use_primitives=use_primitives,
            ),
            compress=True
        )
        # annotation_path = resource_manager.get_file(*mkwad_annotation(seed=0, n_trajs=1000, version="v1"))
        wad = WAD.from_file(self.doom_scenario_path)
        self.wad = wad
        self.level_names = [l.name.encode() for l in wad.levels]
        self.use_primitives = use_primitives
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
        if self.use_primitives:
            doom_config.available_buttons = [
                Button.MOVE_FORWARD_BACKWARD_DELTA,  # SPEED,
                Button.TURN_LEFT_RIGHT_DELTA,  # SPEED,
                # Button.ALTATTACK,
                # Button.RELOAD,
                # Button.USE,
            ]
        else:
            if self.allow_backwards:
                doom_config.available_buttons = [
                    Button.TURN_LEFT,
                    Button.TURN_RIGHT,
                    Button.MOVE_FORWARD,
                    Button.MOVE_BACKWARD,
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
        if self.use_primitives:
            if self.allow_backwards:
                return np.asarray([
                    [0, -1],
                    [0, 1],
                    [1, 0],
                    [-1, 0],
                ], dtype=np.intc)
            else:
                return np.asarray([
                    [0, -1],
                    [0, 1],
                    [1, 0],
                ], dtype=np.intc)
        else:
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
        logger.record_tabular_misc_stat('SuccessTrajLen', success_traj_len)  #

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

    def reset(self, restart_game=None, reset_map=None):
        self.traj = []
        return DoomEnv.reset(self, restart_game=restart_game, reset_map=reset_map)

    def render(self, close=False, wait_key=True, mode='human', full_size=False):
        level_name = self.executor.par_games.get_doom_map(0).decode()
        level = [x for x in self.wad.levels if x.name == level_name][0]
        level_content = [x for x in level.lumps if x.name == "TEXTMAP"][0].content
        agent_x = vizdoom.doom_fixed_to_double(self.executor.par_games.get_game_variable(0, vizdoom.GameVariable.USER1))
        agent_y = vizdoom.doom_fixed_to_double(self.executor.par_games.get_game_variable(0, vizdoom.GameVariable.USER2))
        obs_img = self.get_image_obs(rescale=True, full_size=full_size)
        obs_img = np.cast['uint8']((obs_img + 1) * 0.5 * 255)
        if not full_size:
            obs_img = obs_img.reshape(self.observation_space.shape)
        height, width, _ = obs_img.shape

        from sandbox.rocky.neural_learner.doom_utils.plotting import plot_textmap, parse_map
        parsed_map = parse_map(level_content)

        width, height = 400, 400

        self.traj.append((agent_x, agent_y))

        map_img = plot_textmap(parsed_map, [self.traj], out_width=width, out_height=height,
                               side_length=self.__getstate__()['__args'][2])[0][:, :, ::-1]

        obs_img = cv2.resize(obs_img, (width, height))
        joint_img = np.concatenate([obs_img, map_img], axis=1)
        if mode == 'human':
            cv2.imshow("Map", joint_img)
            if wait_key:
                cv2.waitKey(10)
        elif mode == 'rgb_array':
            return joint_img
        else:
            raise NotImplementedError

    # def render(self, close=False, wait_key=True):
    #     level_name = self.executor.par_games.get_doom_map(0).decode()
    #     level = [x for x in self.wad.levels if x.name == level_name][0]
    #     level_content = [x for x in level.lumps if x.name == "TEXTMAP"][0].content
    #     agent_x = vizdoom.doom_fixed_to_double(self.executor.par_games.get_game_variable(0, vizdoom.GameVariable.USER1))
    #     agent_y = vizdoom.doom_fixed_to_double(self.executor.par_games.get_game_variable(0, vizdoom.GameVariable.USER2))
    #     obs_img = self.get_image_obs(rescale=True)
    #     obs_img = np.cast['uint8']((obs_img + 1) * 0.5 * 255)
    #     obs_img = obs_img.reshape((self.observation_space.shape))
    #     height, width, _ = obs_img.shape
    #     map_img = plot_textmap(level_content, agent_x, agent_y, out_width=400, out_height=400)
    #
    #     obs_img = cv2.resize(obs_img, (400, 400))
    #     joint_img = np.concatenate([obs_img, map_img], axis=1)
    #     cv2.imshow("Map", joint_img)
    #     if wait_key:
    #         cv2.waitKey(10)

    def start_interactive(self):
        self.render(wait_key=False)
        while True:

            up = 63232
            down = 63233
            left = 63234
            right = 63235

            key = int(cv2.waitKey(10))
            if key == left:
                self.step(0)
            elif key == right:
                self.step(1)
            elif key == up:
                self.step(2)
            elif key == down:
                self.step(3)
            else:
                pass
            self.render(wait_key=False)

    # def vec_env_executor(self, n_envs):
    #     return VecGoalFindingDoomEnv(n_envs=n_envs, env=self)

    @property
    def action_space(self):
        return Discrete(len(self.action_map))


class VecGoalFindingDoomEnv(VecDoomEnv):
    def set_action(self, action_n):
        if self.env.use_primitives:
            for i in range(self.n_envs):
                self.par_games.set_action(i, self.env.action_map[action_n[i]])

            import ipdb;
            ipdb.set_trace()
        else:
            VecDoomEnv.set_action(self, action_n)


# def plot_textmap(content, agent_x, agent_y, out_width, out_height, side_length):
#     if isinstance(content, bytes):
#         content = content.decode()
#     parts = content.replace('\n', '').replace(' ', '').split('}')
#
#     parsed_parts = []
#     for part in parts:
#         if len(part) > 0:
#             part = part.split('{')
#             type_part = part[0].split(';')[-1]
#             attrs = dict([x.split('=') for x in part[1].split(';') if len(x) > 0])
#         parsed_parts.append(dict(attrs, klass=type_part))
#
#     vertices = []
#
#     for part in parsed_parts:
#         if part['klass'] == 'vertex':
#             vertices.append((int(part['x']), int(part['y'])))
#
#     vertices = np.asarray(vertices)
#
#     min_x, min_y = np.min(vertices, axis=0)  # - 50
#     max_x, max_y = np.max(vertices, axis=0)  # + 50
#
#     height = max_x - min_x
#     width = max_y - min_y
#     # import ipdb; ipdb.set_trace()
#     # Now plot the map
#
#     import cv2
#     img = np.ones((height, width, 3), dtype=np.uint8) * 255
#
#     def rescale_point(x, y):
#         tx = int((int(x) - min_x) / (max_x - min_x) * height)
#         ty = int((int(y) - min_y) / (max_y - min_y) * width)
#         return (ty, tx)  # tx, ty)
#
#     # cx = width / 2
#     # cy = height / 2
#
#     # cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow("Map", width=1000, height=1000)
#     vertices = []
#     # n_lines = 0
#     # inc = 20  # 5#3#2
#
#     target_coords = []
#
#     for idx, part in enumerate(parsed_parts):
#         if part['klass'] == 'vertex':
#             cv2.circle(img, center=rescale_point(part['x'], part['y']), color=(100, 100, 100), radius=5,
#                        thickness=-1)
#             vertices.append(part)
#             if int(part['x']) % side_length != 0 or int(part['y']) % side_length != 0:
#                 target_coords.append((int(part['x']), int(part['y'])))
#         elif part['klass'] == 'linedef':
#             pt1 = (int(vertices[int(part['v1'])]['x']), int(vertices[int(part['v1'])]['y']))
#             pt2 = (int(vertices[int(part['v2'])]['x']), int(vertices[int(part['v2'])]['y']))
#             if pt1[0] % side_length == 0 and pt1[1] % side_length == 0 and pt2[0] % side_length == 0 and pt2[1] % side_length == 0:
#                 pt1 = rescale_point(*pt1)
#                 pt2 = rescale_point(*pt2)
#                 cv2.line(img, pt1, pt2, color=(0, 0, 0), thickness=5)
#         elif part['klass'] == 'thing':
#             if int(part['type']) == 1:
#                 # character
#                 cv2.circle(img, center=rescale_point(agent_x, agent_y), color=(255, 0, 0), radius=5,
#                            thickness=-1)
#
#     tx, ty = np.mean(target_coords, axis=0)
#
#     cv2.circle(img, center=rescale_point(tx, ty), color=(0, 0, 255), radius=5,
#                thickness=-1)
#
#     img = cv2.resize(img, (out_width, out_height))
#     return img
#
#
# # def plot_textmap(content, agent_x, agent_y, out_width, out_height):
# #     if isinstance(content, bytes):
# #         content = content.decode()
# #     parts = content.replace('\n', '').replace(' ', '').split('}')
# #
# #     parsed_parts = []
# #     for part in parts:
# #         if len(part) > 0:
# #             part = part.split('{')
# #             type_part = part[0].split(';')[-1]
# #             attrs = dict([x.split('=') for x in part[1].split(';') if len(x) > 0])
# #         parsed_parts.append(dict(attrs, klass=type_part))
# #
# #     vertices = []
# #
# #     for part in parsed_parts:
# #         if part['klass'] == 'vertex':
# #             vertices.append((int(part['x']), int(part['y'])))
# #
# #     vertices = np.asarray(vertices)
# #
# #     min_x, min_y = np.min(vertices, axis=0) - 50
# #     max_x, max_y = np.max(vertices, axis=0) + 50
# #
# #     height = max_x - min_x
# #     width = max_y - min_y
# #     # import ipdb; ipdb.set_trace()
# #     # Now plot the map
# #
# #     import cv2
# #     img = np.ones((height, width, 3), dtype=np.uint8) * 255
# #
# #     def rescale_point(x, y):
# #         tx = int((int(x) - min_x) / (max_x - min_x) * height)
# #         ty = int((int(y) - min_y) / (max_y - min_y) * width)
# #         return (ty, tx)  # tx, ty)
# #
# #     # cx = width / 2
# #     # cy = height / 2
# #
# #     # cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
# #     # cv2.resizeWindow("Map", width=1000, height=1000)
# #     vertices = []
# #     # n_lines = 0
# #     # inc = 20  # 5#3#2
# #     for idx, part in enumerate(parsed_parts):
# #         if part['klass'] == 'vertex':
# #             cv2.circle(img, center=rescale_point(part['x'], part['y']), color=(100, 100, 100), radius=5,
# #                        thickness=-1)
# #             vertices.append(part)
# #         elif part['klass'] == 'linedef':
# #             pt1 = rescale_point(vertices[int(part['v1'])]['x'], vertices[int(part['v1'])]['y'])
# #             pt2 = rescale_point(vertices[int(part['v2'])]['x'], vertices[int(part['v2'])]['y'])
# #             cv2.line(img, pt1, pt2, color=(0, 0, 0), thickness=5)
# #         elif part['klass'] == 'thing':
# #             if int(part['type']) == 1:
# #                 # character
# #                 cv2.circle(img, center=rescale_point(agent_x, agent_y), color=(255, 0, 0), radius=5,
# #                            thickness=-1)
# #             elif int(part['type']) == 5:
# #                 cv2.circle(img, center=rescale_point(part['x'], part['y']), color=(0, 0, 255), radius=5,
# #                            thickness=-1)
# #                 # blue keycard
# #
# #     img = cv2.resize(img, (out_width, out_height))
# #     return img
# #
# #
if __name__ == "__main__":
    env = DoomFixedGoalFindingMazeEnv(
        n_trajs=1,
        seed=np.random.randint(100000),
        rescale_obs=None,
        use_primitives=False,#True,
        frame_skip=4,
        allow_backwards=True,#False,
        maze_sizes=[5],
        margin=1,
    )  # use_primitives=True, frame_skip=1)
    env.start_interactive()

    # wad_file = resource_manager.get_file(*mkwad(
    #     n_trajs=2,
    #     version="v2",
    #     side_length=96,
    #     rand_angle=False,
    #     margin=1,
    #     maze_sizes=[3],
    #     wall_penalty=0.001,
    #     seed=np.random.randint(100000),
    #     # map_preset="two_goal_lv2",
    # ), compress=True)
    # class DoomEnv(DoomDefaultWadEnv):
    #
    #     def get_doom_config(self):
    #         cfg = DoomDefaultWadEnv.get_doom_config(self)
    #         cfg.render_weapon = False
    #         cfg.render_hud = False
    #         cfg.render_crosshair = False
    #         cfg.render_decals = False
    #         cfg.render_particles = False
    #         return cfg
    #
    #
    # env = DoomEnv(full_wad_name=wad_file)
    # # env.start_interactive()
    # from vizdoom import Mode, GameVariable
    #
    # # env.mode = Mode.PLAYER  # SPECTATOR
    # env.mode = Mode.SPECTATOR
    # env.executor.init_games()
    # points = []
    # import cv2
    #
    # # cv2.namedWindow("plot")
    # while True:
    #     if env.executor.par_games.is_episode_finished(0):
    #         env.executor.par_games.close_all()
    #         env.executor.init_games()
    #         env.executor.par_games.new_episode_all()
    #     env.executor.par_games.set_action(0, env.action_map[np.random.choice([0, 1, 2, 3])])
    #     env.executor.par_games.advance_action_all(4, update_state=True, render_only=True)
    #     # print(env.executor.par_games.get_total_reward(0))#advance_action_all(4, update_state=True, render_only=True)
    #
    #
    #     # cv2.imshow("Map", img)
    #     # cv2.waitKey(15)
