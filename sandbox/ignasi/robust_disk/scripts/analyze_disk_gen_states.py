import argparse
import csv
import pickle
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.utils import rollout
from sandbox.ignasi.robust_disk.envs.disk_generate_states_env import DiskGenerateStatesEnv
from sandbox.young_clgan.envs.base import FixedStateGenerator
import sys
import scipy.misc
import os

"""
Various utils to inspect/evaluate generated states
"""

NUM_GRID = 5
# put good trained policy below
POLICY_PATH = "data/s3/robust-disk-test/robust-disk-test_2017_08_23_09_58_01_0001/itr_150/params.pkl"
# POLICY_PATH = "data/s3/robust-disk-test/robust-disk-test_2017_08_23_17_06_26_0001/itr_150/params.pkl"

NUM_TIME_STEPS = 500
NUM_STATES = 30


def partition_sampled_states(states, lb, spacing, num_grid=NUM_GRID, transform=lambda x: x[-2:]):
    """
    Given list of states, returns list of lists of states in each grid
    Assumes square big grid
    :return:
    """
    grid_states = []
    for i in range(num_grid):
        to_add = []
        for j in range(num_grid):
            to_add.append([])
        grid_states.append(to_add)
    for i, state in enumerate(states):
        if i % 10000 == 0:
            print(i)
        x, y = transform(state)
        x_index = min(int((x - lb) / spacing), num_grid - 1)
        y_index = min(int((y - lb) / spacing), num_grid - 1)
        grid_states[x_index][y_index].append(state)
    return grid_states


def trim_data_set(data, max_states=2000):
    print("Dividing states into grid spaces")
    grid_states = partition_sampled_states(data, -0.05, 0.02, NUM_GRID)
    trimmed_set = []
    for i in range(NUM_GRID):
        for j in range(NUM_GRID):
            grid = grid_states[i][j]
            num_states_in_grid = len(grid)
            if num_states_in_grid < max_states:
                trimmed_set.extend(grid)
            else:
                indices = np.random.choice(num_states_in_grid, max_states, replace=False)
                # import ipdb; ipdb.set_trace()
                trimmed_set.extend(np.array(grid)[indices].tolist())
    trimmed_set = np.array(trimmed_set)
    folder = "data/trimmed_data_set/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = folder + "all_feasible_states.pkl"
    with open(file, "wb") as f:
        pickle.dump(trimmed_set, f)
    return


def grid_and_analyze_grid(data, rollouts=False, save_images=False, file_name='success_breakdown.csv'):
    # Iterates through points in the grid and performs rollouts to estimate percentage of success

    print("Dividing states into grid spaces")
    grid_states = partition_sampled_states(data, -0.05, 0.02, NUM_GRID)
    save_dir = "data/"  # us.uped to save images of state
    data = joblib.load(POLICY_PATH)
    if "algo" in data:
        policy = data["algo"].policy
        env = data["algo"].env
    else:
        policy = data['policy']
        env = data['env']

    # viewer = env.get_viewer()
    cam_pos = [0, 0.6, 0.5, 0.75, -60, 270]

    with open('data/' + file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["i", "j", "num_states", "success"])
        for i in range(NUM_GRID):
            for j in range(NUM_GRID):
                grid = grid_states[i][j]
                num_states_in_grid = len(grid)
                num_success = 0
                folder = "gen_states_visual/data_{}_{}/".format(i, j)

                # may cause race condition
                if not os.path.exists(folder):
                    os.makedirs(folder)

                if rollouts:
                    indices = np.random.choice(num_states_in_grid, NUM_STATES, replace=False)
                    for n, index in enumerate(indices):
                        sampled_state = grid[index]
                        env.update_start_generator(FixedStateGenerator(sampled_state))

                        # ___CODE FOR ADJUSTING CAMERA ___
                        # xs = [0]
                        # ys = [0.6]
                        # zs= [0.5]
                        # elevations = [-60]
                        # distances = [0.75]
                        # for elevation in elevations:
                        #     for y in ys:
                        #         for x in xs:
                        #             for z in zs:
                        #                 for distance in distances:
                        #                     cam_pos = np.array([x, y, z, distance, elevation, 270.]) # x, y, z, distance, elevation, azimuth
                        #                     env.wrapped_env.setup_camera(cam_pos, viewer)
                        #                     rgb = env.render(mode="rgb_array")
                        #                     scipy.misc.imsave('camera_angle/outfile_{}_{}_{}_{}_{}.jpg'.format(x, y, z, distance, elevation), rgb)
                        # sys.exit(0)
                        if save_images:
                            viewer = env.get_viewer()
                            env.wrapped_env.setup_camera(cam_pos, viewer)
                            rgb = env.render(mode="rgb_array")
                            scipy.misc.imsave(folder + "{}.jpg".format(n), rgb)
                        path = rollout(env, policy, 500, animated=False, speedup=2)
                        success = path["rewards"][-1]
                        num_success += success
                    success_rate = num_success * 1.0 / NUM_STATES
                else:
                    success_rate = -1
                out = i, j, num_states_in_grid, success_rate
                print(out)
                csvwriter.writerow(out)


def plot_peg_position_density(data):
    x_peg = data[:, peg_joints[0]]
    y_peg = data[:, peg_joints[1]]
    fig, ax = plt.subplots()
    heatmap, xedges, yedges = np.histogram2d(x_peg,
                                             y_peg,
                                             bins=[10, 10])
    heatmap /= np.sum(heatmap)
    # Plot peg density
    # eventual todo? we can use grid_states to plot density heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    fig, ax = plt.subplots()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap="Blues")
    plt.colorbar()
    plt.title("Generated peg positions density")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    fig.savefig(args.file + "peg_densities.png")

    # Plots x-y peg positions
    fig, ax = plt.subplots()
    plt.scatter(x_peg, y_peg, c="blue")
    plt.xlim([-0.2, 0.2])
    plt.ylim([-0.2, 0.2])
    plt.title("Generated peg positions")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    fig.savefig(args.file + "peg_positions.png")


def plot_joint_variations(data):
    # Plots variation for each joint
    fig, ax = plt.subplots()
    for joint in range(data.shape[1]):
        if joint in peg_joints:
            plt.scatter([joint] * num_states, data[:, joint] * 100, c="lightgreen")
        else:
            plt.scatter([joint] * num_states, data[:, joint], c="lightgreen")
    plt.scatter(range(9), (0.387, 1.137, -2.028, -1.744, 2.029, -0.873, 1.55, 0, 0), c="black")
    plt.title("Variation for each joint")
    plt.show()
    fig.savefig(args.file + "joint_variations.png")


def show_generated_states(data):
    # Simulates policy to visualize generated states
    env = DiskGenerateStatesEnv(evaluate=True)
    for i in range(100):
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
        )
        rollout(env, policy, animated=True, max_path_length=10, init_state=data[int(random.random() * num_states)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the file with StatesCollection of generated starts')
    args = parser.parse_args()
    # args.file = "/home/michael/rllab_goal_rl/data/local/robust-disk-gen-states-uniform2/robust-disk-gen-states-uniform2_2017_08_20_16_02_56_0001/"
    file = args.file
    all_feasible_starts = pickle.load(open(file, "rb"))

    num_states = all_feasible_starts.size
    print("Getting data from: ", file)
    print("Number of states: ", num_states)
    data = all_feasible_starts.states
    peg_joints = 7, 8

    # trim_data_set(data)
    # grid_and_analyze_grid(data, rollouts=True, save_images=False, file_name="success_breakdown_new.csv")
    #
    print("plotting peg_position_density")
    plot_peg_position_density(data)
    print("plotting joint variations")
    plot_joint_variations(data)
    print("showing generated states")
    show_generated_states(data)
