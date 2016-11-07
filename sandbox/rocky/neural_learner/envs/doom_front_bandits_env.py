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

int reward_means[4] = $reward_means;

script 1 OPEN
{
    reward = 0;
}

script 2 (int bandit_id)
{
    int p = Random(0, 1.0);
    if (p < reward_means[bandit_id]) {
        reward = reward + 1.0;
    } else {
        reward = reward + 0.;
    }
    Exit_Normal(0);
}

script 3 ENTER
{
    while(1)
    {
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
        side_length=200,
        bandit_side_length=100,
        *args, **kwargs):
    from sandbox.rocky.neural_learner.doom_utils.textmap import Thing, Textmap, Vertex, Linedef, Sidedef, Sector

    player = Thing(x=0, y=0, type=1, id=1)

    textmap = Textmap(namespace="zdoom", items=[])

    things = [player]

    # create common components
    textmap.items.extend(things)
    sectors = []
    sectors.append(
        Sector(
            texturefloor="FLAT3",
            textureceiling="FLAT2",
            heightceiling=104,
            lightlevel=210,
        )
    )
    textmap.items.extend(sectors)
    sidedefs = [
        Sidedef(sector=0, texturemiddle="BRICK9"),
        Sidedef(sector=0, texturemiddle="FLOOR1_7", texturetop="FLOOR1_7", clipmidtex=True, scalex_mid=3.0,
                scaley_mid=0.3),
        Sidedef(sector=0, texturemiddle="FLOOR1_1", texturetop="FLOOR1_1", clipmidtex=True, scalex_mid=3.0,
                scaley_mid=0.3),
        Sidedef(sector=0, texturemiddle="RROCK20", texturetop="RROCK20", clipmidtex=True, scalex_mid=3.0,
                scaley_mid=0.3),
        Sidedef(sector=0, texturemiddle="FLAT23", texturetop="FLAT23", clipmidtex=True, scalex_mid=3.0,
                scaley_mid=0.3),
    ]
    textmap.items.extend(sidedefs)

    # create the outermost walls

    vid_start = 0

    if True:
        linedefs = []
        vertices = [
            (-20, -side_length),
            (-20, side_length),
            (side_length, side_length),
            (side_length, -side_length),
        ]
        for vid in range(4):
            vid_next = (vid + 1) % 4
            linedefs.append((vid + vid_start, vid_next + vid_start))
        textmap.items.extend([Vertex(x=v[0], y=v[1]) for v in vertices])
        textmap.items.extend([Linedef(
            v1=l[0], v2=l[1], blocking=True,
            sidefront=0, special=80, arg0=4,
            firstsideonly=True, playerpush=True,
            repeatspecial=True,
        ) for l in linedefs])
        vid_start += 4

    # create the target on the 1st corner

    if True:
        linedefs = []
        vertices = [
            (side_length, side_length),
            (side_length, side_length - bandit_side_length),
            (side_length - bandit_side_length, side_length - bandit_side_length),
            (side_length - bandit_side_length, side_length),
        ]
        # vertices = [
        #     (-side_length, -side_length),
        #     (-side_length, - side_length + bandit_side_length),
        #     (-side_length + bandit_side_length, - side_length + bandit_side_length),
        #     (-side_length + bandit_side_length, - side_length),
        # ]
        for vid in range(4):
            vid_next = (vid + 1) % 4
            linedefs.append((vid_next + vid_start, vid + vid_start))
        textmap.items.extend([Vertex(x=v[0], y=v[1]) for v in vertices])
        textmap.items.extend([Linedef(
            v1=l[0], v2=l[1], blocking=True,
            sidefront=1, special=80, arg0=2,
            firstsideonly=True, playerpush=True,
            repeatspecial=True,
        ) for l in linedefs])
        vid_start += 4

    icr = int((side_length * 2 - bandit_side_length) / 3)

    # create the target on the 2nd corner
    if True:
        linedefs = []
        vertices = [
            (side_length, side_length - icr),
            (side_length, side_length - icr - bandit_side_length),
            (side_length - bandit_side_length, side_length - icr - bandit_side_length),
            (side_length - bandit_side_length, side_length - icr),
        ]
        vertices = [(int(x), int(y)) for x, y in vertices]
        # vertices = [
        #     (-side_length, side_length),
        #     (-side_length, side_length - bandit_side_length),
        #     (-side_length + bandit_side_length, side_length - bandit_side_length),
        #     (-side_length + bandit_side_length, side_length),
        # ]
        for vid in range(4):
            vid_next = (vid + 1) % 4
            linedefs.append((vid_next + vid_start, vid + vid_start))
        textmap.items.extend([Vertex(x=v[0], y=v[1]) for v in vertices])
        textmap.items.extend([Linedef(
            v1=l[0], v2=l[1], blocking=True,
            sidefront=2, special=80, arg0=2,
            firstsideonly=True, playerpush=True,
            repeatspecial=True,
        ) for l in linedefs])
        vid_start += 4

    #
    # create the target on the 3rd corner
    if True:
        linedefs = []
        vertices = [
            (side_length, side_length - 2 * icr),
            (side_length, side_length - 2 * icr - bandit_side_length),
            (side_length - bandit_side_length, side_length - 2 * icr - bandit_side_length),
            (side_length - bandit_side_length, side_length - 2 * icr),
        ]
        vertices = [(int(x), int(y)) for x, y in vertices]
        for vid in range(4):
            vid_next = (vid + 1) % 4
            linedefs.append((vid_next + vid_start, vid + vid_start))
        textmap.items.extend([Vertex(x=v[0], y=v[1]) for v in vertices])
        textmap.items.extend([Linedef(
            v1=l[0], v2=l[1], blocking=True,
            sidefront=3, special=80, arg0=2,
            firstsideonly=True, playerpush=True,
            repeatspecial=True,
        ) for l in linedefs])
        vid_start += 4
    # create the target on the 3rd corner

    if True:
        linedefs = []
        vertices = [
            (side_length, - side_length),
            (side_length, - side_length + bandit_side_length),
            (side_length - bandit_side_length, - side_length + bandit_side_length),
            (side_length - bandit_side_length, - side_length),
        ]
        for vid in range(4):
            vid_next = (vid + 1) % 4
            linedefs.append((vid + vid_start, vid_next + vid_start))
        textmap.items.extend([Vertex(x=v[0], y=v[1]) for v in vertices])
        textmap.items.extend([Linedef(
            v1=l[0], v2=l[1], blocking=True,
            sidefront=4, special=80, arg0=2,
            firstsideonly=True, playerpush=True,
            repeatspecial=True,
        ) for l in linedefs])
        vid_start += 4

    sio = StringIO()
    textmap.write(sio)
    return sio.getvalue()


def mkwad(
        seed=None,
        n_trajs=1000,
        version="v2",
        side_length=200,
        bandit_side_length=50,
        wall_penalty=0,
):
    kwargs = OrderedDict([
        ("n_trajs", n_trajs),
        ("side_length", side_length),
        ("bandit_side_length", bandit_side_length),
        ("seed", seed),
        ("wall_penalty", wall_penalty),
        ("version", version),
    ])

    wad_resource_name = "_".join("{0}_{1}".format(k, v) for k, v in kwargs.items())

    resource_name = "doom/doom_front_bandit/{0}.wad".format(wad_resource_name)

    def mk():
        with using_seed(seed):
            wad = WAD()
            wad.wad_type = "PWAD"

            level_names = ["MAP%02d" % idx for idx in range(n_trajs)]

            for idx, level_name in enumerate(level_names):
                wad.lumps.append(Lump(name=level_name, content=b''))
                map_args = kwargs

                rewards = np.random.uniform(low=0, high=1, size=4)

                map_content = create_map(**map_args)
                wad.lumps.append(Lump(name="TEXTMAP", content=map_content.encode()))

                script = TMPL.substitute(
                    reward_means="{" + ", ".join(["%.4f" % x for x in rewards]) + "}",
                    wall_penalty="%.6f" % wall_penalty,
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


class DoomFrontBanditEnv(DoomEnv, Serializable):
    def __init__(
            self,
            side_length=200,
            bandit_side_length=50,
            n_trajs=1000,
            seed=0,
            allow_backwards=True,
            version="v2",
            wall_penalty=0,
            *args, **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.allow_backwards = allow_backwards
        self.doom_scenario_path = resource_manager.get_file(
            *mkwad(
                seed=seed, n_trajs=n_trajs, version=version, side_length=side_length,
                bandit_side_length=bandit_side_length, wall_penalty=wall_penalty,
            ),
            compress=True
        )
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

        map_img = plot_textmap(level_content, agent_x, agent_y, out_width=400, out_height=400,
                               side_length=self.__getstate__()['__args'][2])

        obs_img = cv2.resize(obs_img, (400, 400))
        joint_img = np.concatenate([obs_img, map_img], axis=1)
        if mode == 'human':
            cv2.imshow("Map", joint_img)
            if wait_key:
                cv2.waitKey(10)
        elif mode == 'rgb_array':
            return joint_img
        else:
            raise NotImplementedError

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

    @property
    def action_space(self):
        return Discrete(len(self.action_map))


def plot_textmap(content, agent_x, agent_y, out_width, out_height, side_length):
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

    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)

    height = max_x - min_x
    width = max_y - min_y
    # Now plot the map

    import cv2
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    def rescale_point(x, y):
        tx = int((int(x) - min_x) / (max_x - min_x) * height)
        ty = int((int(y) - min_y) / (max_y - min_y) * width)
        return (ty, tx)

    vertices = []

    target_coords = []

    for idx, part in enumerate(parsed_parts):
        if part['klass'] == 'vertex':
            cv2.circle(img, center=rescale_point(part['x'], part['y']), color=(100, 100, 100), radius=5,
                       thickness=-1)
            vertices.append(part)
            if int(part['x']) % side_length != 0 or int(part['y']) % side_length != 0:
                target_coords.append((int(part['x']), int(part['y'])))
        elif part['klass'] == 'linedef':
            pt1 = (int(vertices[int(part['v1'])]['x']), int(vertices[int(part['v1'])]['y']))
            pt2 = (int(vertices[int(part['v2'])]['x']), int(vertices[int(part['v2'])]['y']))
            if pt1[0] % side_length == 0 and pt1[1] % side_length == 0 and pt2[0] % side_length == 0 and pt2[
                1] % side_length == 0:
                pt1 = rescale_point(*pt1)
                pt2 = rescale_point(*pt2)
                cv2.line(img, pt1, pt2, color=(0, 0, 0), thickness=5)
        elif part['klass'] == 'thing':
            if int(part['type']) == 1:
                # character
                cv2.circle(img, center=rescale_point(agent_x, agent_y), color=(255, 0, 0), radius=5,
                           thickness=-1)

    tx, ty = np.mean(target_coords, axis=0)

    cv2.circle(img, center=rescale_point(tx, ty), color=(0, 0, 255), radius=5,
               thickness=-1)

    img = cv2.resize(img, (out_width, out_height))
    return img


if __name__ == "__main__":
    wad_file = resource_manager.get_file(*mkwad(
        n_trajs=2,
        version="v2",
        seed=np.random.randint(100000),
    ), compress=True)


    class DoomEnv(DoomDefaultWadEnv):

        def get_doom_config(self):
            cfg = DoomDefaultWadEnv.get_doom_config(self)
            cfg.render_weapon = False
            cfg.render_hud = False
            cfg.render_crosshair = False
            cfg.render_decals = False
            cfg.render_particles = False
            return cfg


    env = DoomEnv(full_wad_name=wad_file)
    from vizdoom import Mode

    env.mode = Mode.SPECTATOR
    env.executor.init_games()
    points = []
    import cv2

    while True:
        if env.executor.par_games.is_episode_finished(0):
            env.executor.par_games.close_all()
            env.executor.init_games()
            env.executor.par_games.new_episode_all()
        env.executor.par_games.set_action(0, env.action_map[np.random.choice([0, 1, 2, 3])])
        env.executor.par_games.advance_action_all(4, update_state=True, render_only=True)
