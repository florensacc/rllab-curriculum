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

NEW_SCRIPT = """
#include "zcommon.acs"

int target_id = 2;
global int 0:reward;
global int 1:x_pos;
global int 2:y_pos;
global int 3:obj_x_pos;
global int 4:obj_y_pos;
global int 5:thing_hidden;

function int distance (int tid1, int tid2)
{
    int x, y, d;
    x = GetActorX(tid1) - GetActorX(tid2) >> 16; // Convert fixed point to integer
    y = GetActorY(tid1) - GetActorY(tid2) >> 16;
    d = sqrt( x*x + y*y );
    return d;
}

function void update_status (void) {
    x_pos = GetActorX(0);
    y_pos = GetActorY(0);
    obj_x_pos = GetActorX(3);
    obj_y_pos = GetActorY(3);
    int dist = distance(0, 3);
    if (dist > %d) {
        if (!thing_hidden) {
            thing_hidden = 1;
            Thing_Move(2, 4, 1);
        }
    } else {
        if (thing_hidden) {
            thing_hidden = 0;
            Thing_Move(2, 3, 1);
        }
    }
}

script 1 OPEN
{
    reward = 0.;
    light_changetovalue(0, %d);
    SetThingSpecial(target_id, ACS_ExecuteAlways, 2);
    thing_hidden = 0;
    update_status();
}

script 2 (void)
{
    reward = reward + 1.0;
    Exit_Normal(0);
}

script 3 ENTER
{
    if (%d) {
        SetActorAngle(0, Random(0, 1.0));
    }
    while(1)
    {
        delay(1);
        update_status();
    }
}
"""


def create_map(size=500, rand_start=False, min_dist=100, scale=4, margin=0, landmarks=False, **kwargs):
    from sandbox.rocky.neural_learner.doom_utils.textmap import Thing, Textmap, Vertex, Linedef, Sidedef, Sector

    textmap = Textmap(namespace="zdoom", items=[])

    # max_dist_sqr = 0
    # best_set = ()
    #
    # for _ in range(1000):

    while True:
        if rand_start:
            sx, sy = np.random.randint(low=-size + margin, high=size - margin, size=2)
        else:
            sx, sy = 0, 0
        gx, gy = np.random.randint(low=-size + margin, high=size - margin, size=2)

        dist_sqr = (sx - gx) ** 2 + (sy - gy) ** 2
        if dist_sqr > min_dist ** 2:
            break
            # if dist_sqr > max_dist_sqr:
            #     max_dist_sqr = dist_sqr
            #     best_set = (sx, sy, gx, gy)

    # sx, sy, gx, gy = 0, 0, 50, 0#best_set

    textmap.items.append(
        Thing(x=int(sx), y=int(sy), type=1, id=1)
    )

    textmap.items.append(
        Thing(x=int(gx), y=int(gy), type=5, id=2, scale=scale)
    )
    # spot for the target
    textmap.items.append(
        Thing(x=int(gx), y=int(gy), type=9001, id=3)
    )
    # spot for some location outside of the map
    textmap.items.append(
        Thing(x=-size * 2, y=-size * 2, type=9001, id=4)
    )
    if landmarks:
        # Add landmarks to make the problem more visually interesting
        textmap.items.append(
            Thing(x=-size + margin, y=-size + margin, type=55, id=5)
        )
        textmap.items.append(
            Thing(x=-size + margin, y=size - margin, type=56, id=6)
        )
        textmap.items.append(
            Thing(x=size - margin, y=size - margin, type=57, id=7)
        )
        textmap.items.append(
            Thing(x=size - margin, y=-size + margin, type=2028, id=8)
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
    textmap.items.append(Sector(texturefloor="FLAT3", textureceiling="FLAT2", heightceiling=104, lightlevel=100, ))

    vertices2 = [
        (40, 40),
        (40, 80),
        (80, 80),
        (80, 40),
    ]
    vertices2 = [Vertex(x=x, y=y) for x, y in vertices2]

    textmap.items.extend(vertices2)

    for idx in range(4):
        textmap.items.append(Linedef(
            v2=idx + 4,
            v1=(idx + 1) % 4 + 4,
            blocking=False,#True,
            sidefront=1,
            sideback=1,
            # dontpegtop=False,#True,
            dontpegbottom=True,#False,#True,
            blockplayers=False,
        ))
    textmap.items.append(Sidedef(sector=0, texturemiddle="FLOOR1_7", texturetop="FLOOR1_7", clipmidtex=True,
                                 scalex_mid=3.0, scaley_mid=3.0))
    # textmap.items.append(Sidedef(sector=1, texturemiddle="BRICK9", clipmidtex=True))
    # textmap.items.append(Sector(texturefloor="FLAT3", textureceiling="FLAT2", heightceiling=0, lightlevel=210))


    sio = StringIO()
    textmap.write(sio)
    content = sio.getvalue()
    return content


def mkwad(size=500, rand_start=False, min_dist=100, scale=4, light_level=110, seed=0, n_trajs=1000, version="v1",
          forced_visible_range=None, margin=0, landmarks=False, rand_angle=False, verbose=False):
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
        ("forced_visible_range", forced_visible_range),
        ("margin", margin),
        ("landmarks", landmarks),
        ("rand_angle", rand_angle),
        ("version", version),
    ])

    wad_resource_name = "_".join("{0}_{1}".format(k, v) for k, v in kwargs.items())

    resource_name = "doom/doom_po/{0}.wad".format(wad_resource_name)

    def mk():
        if verbose:
            print("Generating %s" % resource_name)

        with using_seed(seed):
            if forced_visible_range is None:
                script = (SCRIPT % light_level).encode()
            else:
                script = (NEW_SCRIPT % (int(forced_visible_range), light_level, int(rand_angle))).encode()

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

                ACTOR DoomPlayer1 : DoomPlayer replaces DoomPlayer
                {
                    Speed 10
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
        size=200,  # 500,
        rand_start=False,
        n_trajs=2,
        light_level=80,#255,#80,  # 255,
        version="v2",
        forced_visible_range=400,#150,
        scale=1,
        margin=20,
        landmarks=True,
        verbose=True,
        rand_angle=False,#True,
        seed=np.random.randint(100000)
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
        obj_x = vizdoom.doom_fixed_to_double(env.executor.par_games.get_game_variable(0, GameVariable.USER3))
        obj_y = vizdoom.doom_fixed_to_double(env.executor.par_games.get_game_variable(0, GameVariable.USER4))

        # print(agent_x, agent_y, obj_x, obj_y)
        # print(env.executor.par_games.get_total_reward(0))
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
