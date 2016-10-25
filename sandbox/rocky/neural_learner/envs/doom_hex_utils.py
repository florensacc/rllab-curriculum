import io
from collections import OrderedDict
from io import StringIO

from rllab.misc.ext import using_seed
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
    SetThingSpecial(target_id, ACS_ExecuteAlways, 2);
}


script 2 (void)
{
    reward = reward + %.5f;
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


def plot_textmap(content, width=1000, height=1000):
    parts = content.replace('\n', '').replace(' ', '').split('}')

    parsed_parts = []
    for part in parts:
        if len(part) > 0:
            part = part.split('{')
            type_part = part[0].split(';')[-1]
            attrs = dict([x.split('=') for x in part[1].split(';') if len(x) > 0])
        parsed_parts.append(dict(attrs, klass=type_part))

    # Now plot the map

    import cv2
    img = np.ones((2000, 2000, 3), dtype=np.uint8) * 255

    # cx = width / 2
    # cy = height / 2

    # cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Map", width=1000, height=1000)
    vertices = []
    # n_lines = 0
    # inc = 20  # 5#3#2
    for idx, part in enumerate(parsed_parts):
        if part['klass'] == 'vertex':
            cv2.circle(img, center=(int(part['x']), int(part['y'])), color=(100, 100, 100), radius=5,
                       thickness=-1)
            # cv2.circle(img, center=(int(part['x']), int(part['y'])), color=(0, 0, 0), radius=5, thickness=-1)
            vertices.append(part)
        elif part['klass'] == 'linedef':
            pt1 = (int(vertices[int(part['v1'])]['x']), int(vertices[int(part['v1'])]['y']))
            pt2 = (int(vertices[int(part['v2'])]['x']), int(vertices[int(part['v2'])]['y']))
            cv2.line(img, pt1, pt2, color=(0, 0, 0), thickness=5)
        elif part['klass'] == 'thing':
            if int(part['type']) == 1:
                # character
                cv2.circle(img, center=(int(part['x']), int(part['y'])), color=(255, 0, 0), radius=10,
                           thickness=-1)
            elif int(part['type']) == 5:
                cv2.circle(img, center=(int(part['x']), int(part['y'])), color=(0, 0, 255), radius=10,
                           thickness=-1)
                # blue keycard

    img = cv2.resize(img, (600, 600))

    cv2.imshow("Map", img)
    cv2.waitKey(0)


def create_map(
        side_length=120,
        margin=60,
        offset_x=1000,
        offset_y=1000,
        tolerance=20,
        n_repeats=1,
        n_targets=1,
        # If difficulty is positive, this number of agent - target pairs will be sampled, and the one with the maximal
        # distance will be chosen; if it is negative, the min distance instead of max distance is used
        difficulty=1,  # 0
        scale=1,
        randomize_texture=True,
        *args,
        **kwargs
):
    assert difficulty != 0
    from sandbox.rocky.neural_learner.doom_utils.textmap import Thing, Textmap, Vertex, Linedef, Sidedef, Sector
    flat_textures = [
        "FLAT1",
        "FLAT1_1",
        "FLAT1_2",
        "FLAT1_3",
        "FLAT2",
        "FLAT3",
        "FLAT4",
        "FLAT5",
        "FLAT5_1",
        "FLAT5_2",
        "FLAT5_3",
        "FLAT5_4",
        "FLAT5_5",
        # "FLAT5_6", # skulls...
        "FLAT5_7",
        "FLAT5_8",
        "FLAT5_9",
        "FLAT8",
        "FLAT9",
        "FLAT10",
        "FLAT14",
        "FLAT17",
        "FLAT18",
        "FLAT19",
        "FLAT20",
        "FLAT22",
        "FLAT23",
    ]

    rrock_textures = [
        "RROCK01",
        "RROCK02",
        "RROCK03",
        "RROCK04",
        "RROCK05",
        "RROCK06",
        "RROCK07",
        "RROCK08",
        "RROCK09",
        "RROCK10",
        "RROCK11",
        "RROCK12",
        "RROCK13",
        "RROCK14",
        "RROCK15",
        "RROCK16",
        "RROCK17",
        "RROCK18",
        "RROCK19",
        "RROCK20",
    ]

    slime_textures = [

        "SLIME01",
        "SLIME02",
        "SLIME03",
        "SLIME04",
        "SLIME05",
        "SLIME06",
        "SLIME07",
        "SLIME08",
        "SLIME09",
        "SLIME10",
        "SLIME11",
        "SLIME12",
        "SLIME13",
        "SLIME14",
        "SLIME15",
        "SLIME16",
    ]

    floor_textures = [

        "FLOOR0_1",
        "FLOOR0_2",
        "FLOOR0_3",
        "FLOOR0_5",
        "FLOOR0_6",
        "FLOOR0_7",
        "FLOOR1_1",
        "FLOOR1_6",
        "FLOOR1_7",
        "FLOOR3_3",
        "FLOOR4_1",
        "FLOOR4_5",
        "FLOOR4_6",
        "FLOOR4_8",
        "FLOOR5_1",
        "FLOOR5_2",
        "FLOOR5_3",
        "FLOOR5_4",
        "FLOOR6_1",
        "FLOOR6_2",
        "FLOOR7_1",
        "FLOOR7_2",
    ]
    ceil_textures = [
        "CEIL1_1",
        "CEIL1_2",
        "CEIL1_3",
        "CEIL3_1",
        "CEIL3_2",
        "CEIL3_3",
        "CEIL3_4",
        "CEIL3_5",
        "CEIL3_6",
        "CEIL4_1",
        "CEIL4_2",
        "CEIL4_3",
        "CEIL5_1",
        "CEIL5_2",

    ]

    all_textures = ceil_textures + rrock_textures + floor_textures + slime_textures + flat_textures

    textmap = Textmap(namespace="zdoom", items=[])
    vertices = []
    linedefs = []
    if randomize_texture:
        sidedefs = [
            Sidedef(sector=0, texturemiddle=random.choice(all_textures))
            for _ in range(100)
            ]
        sectors = [
            Sector(
                texturefloor=random.choice(all_textures),
                textureceiling=random.choice(all_textures),
                heightceiling=104,
                lightlevel=210,
            )
        ]
    else:
        sidedefs = [Sidedef(sector=0, texturemiddle="BRICK9")]
        sectors = [
            Sector(
                texturefloor="FLAT3",
                textureceiling="FLAT2",
                heightceiling=104,
                lightlevel=210,
            )
        ]

    hexagonals = []

    def draw_hexagonal(center, side_length):
        cx, cy = center
        vs = [
            # top right vertex
            (cx + int(0.5 * side_length), cy + int((3 ** 0.5) * 0.5 * side_length)),
            # right vertex
            (cx + int(side_length), cy + int(0)),
            # bottom right vertex
            (cx + int(0.5 * side_length), cy + int(-(3 ** 0.5) * 0.5 * side_length)),
            # bottom left vertex
            (cx + int(-0.5 * side_length), cy + int(-(3 ** 0.5) * 0.5 * side_length)),
            # left vertex
            (cx + int(-side_length), cy + int(0)),
            # top left vertex
            (cx + int(-0.5 * side_length), cy + int((3 ** 0.5) * 0.5 * side_length)),
        ]

        hexagonals.append(vs)

        vs = vs[::-1]

        vertices.extend(vs)

        side = random.randint(0, len(sidedefs) - 1)

        for idx in range(6):
            linedefs.append(Linedef(
                v1=vertices.index(vs[idx]),
                v2=vertices.index(vs[(idx + 1) % 6]),
                blocking=True,
                sidefront=side
            ))

    def get_top(ptx, side_length, margin):
        cx, cy = ptx
        return (
            cx,
            int(cy - side_length * (3 ** 0.5) - margin)
        )

    def get_top_right(ptx, side_length, margin):
        cx, cy = ptx
        return (
            int(cx + 1.5 * side_length + (3 ** 0.5) * 0.5 * margin),
            int(cy - 0.5 * side_length * (3 ** 0.5) - 0.5 * margin)
        )

    def get_top_left(ptx, side_length, margin):
        cx, cy = ptx
        return (
            int(cx - 1.5 * side_length - (3 ** 0.5) * 0.5 * margin),
            int(cy - 0.5 * side_length * (3 ** 0.5) - 0.5 * margin)
        )

    def get_bottom_left(ptx, side_length, margin):
        cx, cy = ptx
        return (
            int(cx - 1.5 * side_length - (3 ** 0.5) * 0.5 * margin),
            int(cy + 0.5 * side_length * (3 ** 0.5) + 0.5 * margin)
        )

    def get_bottom_right(ptx, side_length, margin):
        cx, cy = ptx
        return (
            int(cx + 1.5 * side_length + (3 ** 0.5) * 0.5 * margin),
            int(cy + 0.5 * side_length * (3 ** 0.5) + 0.5 * margin)
        )

    def get_bottom(ptx, side_length, margin):
        cx, cy = ptx
        return (
            cx,
            int(cy + side_length * (3 ** 0.5) + margin)
        )

    def has_dup(ptx, ptx_list):
        for ptx1 in ptx_list:
            dist = abs(ptx[0] - ptx1[0]) + abs(ptx[1] - ptx1[1])
            if dist <= 6:
                return True
        return False

    ptx = (offset_x, offset_y)

    pts = [ptx]
    drawn = []

    directions = [get_top, get_top_left, get_top_right, get_bottom, get_bottom_left, get_bottom_right]

    for _ in range(n_repeats):

        for ptx in list(pts):

            if not has_dup(ptx, drawn):
                draw_hexagonal(ptx, side_length=side_length)
                drawn.append(ptx)

            for dir in directions:
                ptx1 = dir(ptx, side_length=side_length, margin=margin)
                if not has_dup(ptx1, pts):
                    pts.append(ptx1)
                if not has_dup(ptx1, drawn):
                    draw_hexagonal(ptx1, side_length=side_length)
                    drawn.append(ptx1)

    # Now draw the outer enclosing hexagonal

    outer_side_length = int(0.5 * (3 ** 0.5) * (2 * n_repeats + 1) * side_length + margin * (n_repeats + 1.5))

    vs = [
             (offset_x, offset_y - outer_side_length),
             (offset_x + 0.5 * (3 ** 0.5) * outer_side_length, offset_y - 0.5 * outer_side_length),
             (offset_x + 0.5 * (3 ** 0.5) * outer_side_length, offset_y + 0.5 * outer_side_length),
             (offset_x, offset_y + outer_side_length),
             (offset_x - 0.5 * (3 ** 0.5) * outer_side_length, offset_y + 0.5 * outer_side_length),
             (offset_x - 0.5 * (3 ** 0.5) * outer_side_length, offset_y - 0.5 * outer_side_length),
         ][::-1]
    vs = [(int(v[0]), int(v[1])) for v in vs]

    outer_hexagonal = vs

    vertices.extend(vs)

    side = random.randint(0, len(sidedefs) - 1)
    for idx in range(6):
        linedefs.append(Linedef(
            v1=vertices.index(vs[idx]),
            v2=vertices.index(vs[(idx + 1) % 6]),
            blocking=True,
            sidefront=side
        ))

    vertices = [Vertex(x=x, y=y) for x, y in vertices]

    textmap.items.extend(vertices)
    textmap.items.extend(linedefs)
    textmap.items.extend(sidedefs)
    textmap.items.extend(sectors)

    def is_in_outer_hexagonal(ptx, hexagonal, prefer_in=None, tol=20):
        px, py = ptx
        cx, cy = np.mean(hexagonal, axis=0)
        side_length = np.mean(np.linalg.norm(np.asarray(hexagonal) - np.mean(hexagonal, axis=0, keepdims=True),
                                             axis=1))
        if prefer_in == True:
            side_length = side_length + tol
        elif prefer_in == False:
            side_length = side_length - tol
        h = side_length * 0.5 * (3 ** 0.5)
        v = side_length * 0.5
        q2x = np.abs(px - cx)
        q2y = np.abs(py - cy)

        if q2x > h or q2y > v * 2:
            return False
        return 2 * v * h - v * q2x - h * q2y >= 0

    def is_in_inner_hexagonal(ptx, hexagonal, prefer_in=None, tol=20):
        px, py = ptx
        # rotate the point by 30 degrees
        cx, cy = np.mean(hexagonal, axis=0)
        side_length = np.mean(np.linalg.norm(np.asarray(hexagonal) - np.mean(hexagonal, axis=0, keepdims=True),
                                             axis=1))
        if prefer_in == True:
            side_length = side_length + tol
        elif prefer_in == False:
            side_length = side_length - tol
        h = side_length * 0.5
        v = side_length * 0.5 * (3 ** 0.5)
        q2x = np.abs(px - cx)
        q2y = np.abs(py - cy)

        if q2x > h * 2 or q2y > v:
            return False
        return 2 * v * h - v * q2x - h * q2y >= 0

    opt_dist = None

    for _ in range(abs(difficulty)):
        cur_pts = []

        for idxx in range(1 + n_targets):

            while True:
                max_x = np.max([v.x for v in vertices])
                max_y = np.max([v.y for v in vertices])
                min_x = np.min([v.x for v in vertices])
                min_y = np.min([v.y for v in vertices])

                rand_x = np.random.randint(low=min_x, high=max_x)
                rand_y = np.random.randint(low=min_y, high=max_y)

                pt = (rand_x, rand_y)
                if is_in_outer_hexagonal(pt, outer_hexagonal, prefer_in=False, tol=side_length) and not \
                        any(is_in_inner_hexagonal(pt, hex, prefer_in=True, tol=tolerance) for hex in hexagonals):
                    break
            cur_pts.append(pt)
        agent = cur_pts[0]
        counter_opt_dist = None
        for target in cur_pts[1:]:
            dist = (agent[0] - target[0]) ** 2 + (agent[1] - target[1]) ** 2
            if counter_opt_dist is None or (difficulty > 0 and dist < counter_opt_dist) or (difficulty < 0 and dist >
                counter_opt_dist):
                counter_opt_dist = dist
        if opt_dist is None or (difficulty > 0 and counter_opt_dist > opt_dist) or (difficulty < 0 and
                                                                                            counter_opt_dist < opt_dist):
            all_pts = cur_pts
            opt_dist = counter_opt_dist

    textmap.items.append(
        Thing(x=int(all_pts[0][0]), y=int(all_pts[0][1]), type=1, id=1)
    )
    for target in all_pts[1:]:
        textmap.items.append(
            Thing(x=int(target[0]), y=int(target[1]), type=5, id=2, scale=scale)
        )

    sio = StringIO()
    textmap.write(sio)
    content = sio.getvalue()
    return content


def mkwad(seed, n_trajs, version, side_length=120, margin=60, offset_x=1000, offset_y=1000, tolerance=20, n_repeats=2,
          # If difficulty is positive, this number of agent - target pairs will be sampled, and the one with the maximal
          # distance will be chosen; if it is negative, the min distance instead of max distance is used
          difficulty=1, alive_reward=1.0, randomize_texture=True, n_targets=1, scale=1, verbose=False):
    from sandbox.rocky.neural_learner.doom_utils.wad import Lump, compile_script
    from sandbox.rocky.neural_learner.doom_utils.wad import WAD
    kwargs = OrderedDict([
        ("n_trajs", n_trajs),
        ("side_length", side_length),
        ("margin", margin),
        ("offset_x", offset_x),
        ("offset_y", offset_y),
        ("tolerance", tolerance),
        ("n_repeats", n_repeats),
        ("difficulty", difficulty),
        ("version", version),
        ("seed", seed),
        ("alive_reward", alive_reward),
        ("randomize_texture", randomize_texture),
        ("scale", scale),
        ("n_targets", n_targets),
    ])

    wad_resource_name = "_".join("{0}_{1}".format(k, v) for k, v in kwargs.items())

    resource_name = "doom/hex_goal_finding_maze/{0}.wad".format(wad_resource_name)

    def mk():

        if verbose:
            print("Generating %s" % resource_name)
        with using_seed(seed):
            script = (SCRIPT % alive_reward).encode()

            behavior = compile_script(script)

            wad = WAD()
            wad.wad_type = "PWAD"

            level_names = ["MAP%02d" % idx for idx in range(n_trajs)]

            for level_name in level_names:
                if verbose:
                    print(level_name)
                wad.lumps.append(Lump(name=level_name, content=b''))
                wad.lumps.append(Lump(name="TEXTMAP", content=create_map(**kwargs).encode()))
                wad.lumps.append(Lump(name="SCRIPT", content=script))
                wad.lumps.append(Lump(name="BEHAVIOR", content=behavior))
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
    # from rllab.misc.instrument import run_experiment_lite
    # map = create_map(n_targets=5, difficulty=5)
    # plot_textmap(map)
    def run_task(_):
        for seed in [0, 1, 2, 3, 4]:
            for n_trajs in [10]:#100, 1000, 10000]:
                for difficulty in [1]:#range(-5, 6):
                    if difficulty != 0:
                        file = resource_manager.get_file(
                            *mkwad(
                                seed=seed,
                                n_trajs=n_trajs,
                                version="v1",
                                n_repeats=1,
                                difficulty=difficulty,
                                verbose=True,
                                alive_reward=1.0,
                                n_targets=5,
                            ),
                            compress=True
                        )
                        # print(file)
                        sys.exit()


    run_task(None)
    # run_experiment_lite(
    #     run_task,
    #     mode="local_docker",
    #     use_cloudpickle=True,
    #     docker_image="dementrock/rllab3-vizdoom-gpu-cuda80:cig"
    # )
