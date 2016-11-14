from rllab.misc.ext import using_seed
from sandbox.rocky.s3.resource_manager import resource_manager
import tensorflow as tf
import joblib
import vizdoom
import numpy as np
import cv2
import gizeh
import os
from rllab import config
from rllab.misc import console
from pyearth import Earth
import math

resource_name = "saved_params/doom-maze/doom-maze-77_2016_10_30_22_11_44_0012.pkl"
# resource_name = "saved_params/doom-maze/doom-maze-77_2016_10_30_22_11_44_0004.pkl"
file_name = resource_manager.get_file(resource_name)

save_folder = os.path.join(config.PROJECT_PATH, "data/images/doom-maze")

console.mkdir_p(save_folder)



def lineMagnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude


# Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
def DistancePointLine(px, py, x1, y1, x2, y2):
    # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        # // closest point does not fall within the line segment, take the shorter distance
        # // to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            return (x2, y2)
        else:
            return (x1, y1)
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        return (ix, iy)


with tf.Session() as sess:
    data = joblib.load(file_name)

    env = data['env']
    doom_env = env.wrapped_env.wrapped_env

    policy = data['policy']

    side_length = doom_env.__getstate__()['__args'][2]

    for seed in range(1000):

        # seed = 1

        with using_seed(seed):

            executor = doom_env.executor

            obs = env.reset()

            level_name = executor.par_games.get_doom_map(0).decode()
            level = [x for x in doom_env.wad.levels if x.name == level_name][0]
            level_content = [x for x in level.lumps if x.name == "TEXTMAP"][0].content

            parsed_map = parse_map(level_content)
            tx, ty = get_target_coords(parsed_map)

            done = False

            policy.reset()

            trajs = []

            traj = []

            while not done:
                agent_x = vizdoom.doom_fixed_to_double(
                    executor.par_games.get_game_variable(0, vizdoom.GameVariable.USER1))
                agent_y = vizdoom.doom_fixed_to_double(
                    executor.par_games.get_game_variable(0, vizdoom.GameVariable.USER2))

                traj.append((agent_x, agent_y))

                action, _ = policy.get_action(obs)
                next_obs, rew, done, info = env.step(action)
                obs = next_obs

                if info['episode_done']:
                    # instead of appending that point, we append the point that's closest to the target region
                    if rew > 0:
                        margin = 10
                        cur_pt = traj[-1]
                        segs = [
                            (tx - side_length / 2 + margin, ty - side_length / 2 + margin),
                            (tx - side_length / 2 + margin, ty + side_length / 2 - margin),
                            (tx + side_length / 2 - margin, ty + side_length / 2 - margin),
                            (tx + side_length / 2 - margin, ty - side_length / 2 + margin),
                        ]
                        best_dist = None
                        best_pt = None
                        for pta, ptb in zip(segs, segs[1:] + segs[:1]):
                            closest_pt = DistancePointLine(*cur_pt, *pta, *ptb)
                            dist = np.linalg.norm(np.asarray(closest_pt) - np.asarray(cur_pt))
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                best_pt = closest_pt
                        traj.append(best_pt)#(tx, ty))
                    trajs.append(traj)
                    traj = []

            # trajs.append(traj)

            len1 = len(trajs[0])
            len2 = len(trajs[1])

            diff = len1 - len2

            for traj_idx, traj in enumerate(trajs):

                img, surface = plot_textmap(parsed_map, [traj], out_width=400, out_height=400, side_length=side_length)

                if diff > 0:
                    file_name = "good_{diff}_seed_{seed}_traj_{idx}.png".format(diff=abs(diff), seed=seed, idx=traj_idx)
                elif diff == 0:
                    file_name = "neutral_seed_{seed}_traj_{idx}.png".format(seed=seed, idx=traj_idx)
                else:
                    file_name = "bad_{diff}_seed_{seed}_traj_{idx}.png".format(diff=abs(diff), seed=seed, idx=traj_idx)

                surface.write_to_png(os.path.join(save_folder, file_name))

                print(file_name)

                # cv2.imshow("Image", img)
                # cv2.waitKey()

                # break
