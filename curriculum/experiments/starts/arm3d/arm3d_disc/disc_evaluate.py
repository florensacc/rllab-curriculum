import os.path as osp
import argparse
import pickle
import joblib
import tensorflow as tf

from rllab.sampler.utils import rollout
from rllab.misc.ext import set_seed
from curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv
from curriculum.envs.base import FixedStateGenerator
import numpy as np
import csv

def disc_evaluate(env, policy, init_state = None, max_path_length = 2000, animated = True, speedup = 2, num_rollouts = 50):
    mean_distance = []
    mean_success = []
    for i in range(num_rollouts):
        path = rollout(env, policy, max_path_length=max_path_length,
                       animated=animated, speedup=speedup)
        final_distance = path['env_infos']['distance'][-1]
        success = path["rewards"][-1]
        # print("Trajectory length: {}".format(len(path["rewards"])))
        # print("Success: {}".format(success))
        # print("Final distance: {}". format(final_distance))
        mean_distance.append(final_distance)
        mean_success.append(success)
    return mean_distance, mean_success

def hack_line(x, y, file_name = "/home/michael/rllab_goal_rl/vendor/mujoco_models/arm3d_disc.xml",
    ):
    # Modifies peg position on the xml file
    # todo: make path relative
    # todo: should probably write to
    # tmp folder
    f = open(file_name, 'r')
    lines = f.readlines()
    line = '            <body name="peg" pos="{} {} 0.0">\n'.format(x, y)
    lines[88] = line # super hacky!
    f.close()

    f = open(file_name, 'w')
    f.writelines(lines)
    f.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('file', type=str,
    #                     help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Fixed random seed')
    parser.add_argument("-is", '--init_state', type=str,
                        help='vector of init_state')
    parser.add_argument("-cf", '--collection_file', type=str,
                        help='path to the pkl file with start positions Collection')
    args = parser.parse_args()

    policy = None
    env = None

    args.file = "/home/michael/rllab_goal_rl/disc_best_params.pkl"
    args.speedup = 10
    if args.seed >= 0:
        set_seed(args.seed)
    if args.collection_file:
        all_feasible_starts = pickle.load(open(args.collection_file, 'rb'))

    with tf.Session() as sess:
        data = joblib.load(args.file)
        if "algo" in data:
            policy = data["algo"].policy
            env = data["algo"].env
        else:
            policy = data['policy']
            env = data['env']

        # parameters
        env.terminal_eps = 0.05 # threshold for finishing, bigger to make easier
        length = 5
        visualize = False

        delta_x = [i * 0.01 for i in range(-length, length + 1)]
        delta_y = [i * 0.01 for i in range(-length, length + 1)]

        with open('disc_benchmark.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["x", "y", "distance", "success"])

            for x in delta_x:
                for y in delta_y:
                    hack_line(x, y)
                    env._wrapped_env._wrapped_env = Arm3dDiscEnv() # hopefully this reinitializes
                    # env = data["algo"].env # reloads with new xml file? nope! if not, then check
                    # env.wrapped env something
                    mean_distance, mean_success = disc_evaluate(env, policy, init_state=(x, y),
                                         max_path_length=args.max_path_length,
                                         animated=visualize, speedup=args.speedup)
                    print("x: {}  y: {} distance: {}, success: {}".format(x, y, np.mean(mean_distance), np.mean(mean_success)))
                    csvwriter.writerow([x, y, np.mean(mean_distance), np.mean(mean_success)])

