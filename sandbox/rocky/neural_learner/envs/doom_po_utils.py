import io
from collections import OrderedDict
from io import StringIO

import vizdoom

from rllab.misc.ext import using_seed
from sandbox.rocky.neural_learner.envs.doom_default_wad_env import DoomDefaultWadEnv
from sandbox.rocky.s3.resource_manager import resource_manager

import numpy as np
import random

SCRIPT = """
#include "zcommon.acs"

int target_id = 2;
global int 0:reward;
global int 1:x_pos;
global int 2:y_pos;


script 1 OPEN
{
    reward = 0;
    light_changetovalue(0, %d);
    SetThingSpecial(target_id, ACS_ExecuteAlways, 2);
}

script 2 (void)
{
    reward = reward + 1.0;
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


def create_map(size=500, rand_start=False, min_dist=100, version="v1", scale=4, **kwargs):
    from sandbox.rocky.neural_learner.doom_utils.textmap import Thing, Textmap, Vertex, Linedef, Sidedef, Sector

    textmap = Textmap(namespace="zdoom", items=[])

    while True:
        if rand_start:
            sx, sy = np.random.randint(low=-size, high=size, size=2)
        else:
            sx, sy = 0, 0
        gx, gy = np.random.randint(low=-size, high=size, size=2)

        dist_sqr = (sx - gx) ** 2 + (sy - gy) ** 2
        if dist_sqr > min_dist ** 2:
            break

    textmap.items.append(
        Thing(x=int(sx), y=int(sy), type=1, id=1)
    )
    textmap.items.append(
        Thing(x=int(gx), y=int(gy), type=5, id=2, scale=scale)
    )

    vertices = [
        (-size, -size),
        (-size, size),
        (size, size),
        (size, -size),
    ]

    vertices = [Vertex(x=x, y=y) for x, y in vertices]

    textmap.items.extend(vertices)

    for idx in range(4):
        textmap.items.append(Linedef(
            v1=idx,
            v2=(idx + 1) % 4,
            blocking=True,
            sidefront=0,
        ))

    textmap.items.append(Sidedef(sector=0, texturemiddle="BRICK9"))
    textmap.items.append(Sector(texturefloor="FLAT3", textureceiling="FLAT2", heightceiling=104, lightlevel=210, ))

    sio = StringIO()
    textmap.write(sio)
    content = sio.getvalue()
    return content


def mkwad(size=500, rand_start=False, min_dist=100, scale=4, light_level=110, seed=0, n_trajs=1000, version="v1",
          verbose=False):
    from sandbox.rocky.neural_learner.doom_utils.wad import Lump, compile_script
    from sandbox.rocky.neural_learner.doom_utils.wad import WAD

    kwargs = OrderedDict([
        ("n_trajs", n_trajs),
        ("size", size),
        ("rand_start", rand_start),
        ("min_dist", min_dist),
        ("scale", scale),
        ("version", version),
        ("seed", seed),
        ("light_level", light_level),
        ("n_trajs", n_trajs),
        ("version", version),
    ])

    wad_resource_name = "_".join("{0}_{1}".format(k, v) for k, v in kwargs.items())

    resource_name = "doom/doom_po/{0}.wad".format(wad_resource_name)

    def mk():
        if verbose:
            print("Generating %s" % resource_name)

        with using_seed(seed):
            script = (SCRIPT % light_level).encode()

            behavior = compile_script(script)

            wad = WAD()
            wad.wad_type = "PWAD"

            level_names = ["MAP%02d" % idx for idx in range(n_trajs)]

            # disable the glowing effect of the blue key card
            decorate = """
                ACTOR bluecard1 : BlueCard replaces BlueCard
                {
                  States
                  {
                  Spawn:
                    BKEY A 10
                    Loop
                  }
                }
            """.encode()

            for level_name in level_names:
                if verbose:
                    print(level_name)
                wad.lumps.append(Lump(name=level_name, content=b''))
                wad.lumps.append(Lump(name="TEXTMAP", content=create_map(**kwargs).encode()))
                wad.lumps.append(Lump(name="SCRIPT", content=script))
                wad.lumps.append(Lump(name="BEHAVIOR", content=behavior))
                wad.lumps.append(Lump(name="DECORATE", content=decorate))
                wad.lumps.append(Lump(name="ENDMAP", content=b''))

            wad.reorganize()

            bio = io.BytesIO()

            wad.save_io(bio)

            bytes = bio.getvalue()

            if verbose:
                print("uploading...")
            resource_manager.register_data(resource_name, bytes, compress=True)
            if verbose:
                print("uploaded")

    return resource_name, mk


if __name__ == "__main__":
    wad_file = resource_manager.get_file(*mkwad(
        size=500,
        rand_start=False,
        n_trajs=1000,
        version="v1",
        verbose=True
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
    # env.start_interactive()
    from vizdoom import Mode, GameVariable

    # env.mode = Mode.PLAYER  # SPECTATOR
    env.mode = Mode.SPECTATOR
    env.executor.init_games()
    points = []
    import cv2

    # cv2.namedWindow("plot")
    while True:
        if env.executor.par_games.is_episode_finished(0):
            env.executor.par_games.close_all()
            env.executor.init_games()
            env.executor.par_games.new_episode_all()
        env.executor.par_games.set_action(0, env.action_map[np.random.choice([0, 1, 2, 3])])
        env.executor.par_games.advance_action_all(4, update_state=True, render_only=True)

        # import time
        #
        # time.sleep(0.028)

        agent_x = vizdoom.doom_fixed_to_double(env.executor.par_games.get_game_variable(0, GameVariable.USER1))
        agent_y = vizdoom.doom_fixed_to_double(env.executor.par_games.get_game_variable(0, GameVariable.USER2))

        print(agent_x, agent_y)
        print(env.executor.par_games.get_total_reward(0))
        continue

        points.append((agent_x, agent_y))

        if random.randint(0, 10) == 0:

            # print(agent_x, agent_y)

            img = np.ones((500, 500, 3), dtype=np.uint8) * 255  # zeros

            for x, y in points:
                cv2.circle(img, center=(int(x / (SIZE / 250) + 250), int(y / (SIZE / 250) + 250)), radius=5,
                           color=(0, 0,
                                  0))

            cv2.imshow("image", img)
            cv2.waitKey(10)
