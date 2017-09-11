import argparse
import csv
import pickle
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.utils import rollout
from sandbox.young_clgan.robust_disk.envs.disk_generate_states_env_old import DiskGenerateStatesEnv
from sandbox.young_clgan.envs.base import FixedStateGenerator
import sys
import scipy.misc
import os

from PIL import Image
# import scipy.misc
from sandbox.young_clgan.state.utils import StateCollection

"""
Various utils to inspect/evaluate generated states
"""

NUM_GRID = 5
# put good trained policy
POLICY_PATH = None
# POLICY_PATH = "data/s3/robust-disk-test/robust-disk-test_2017_08_23_09_58_01_0001/itr_150/params.pkl"
# POLICY_PATH = "data/s3/robust-disk-test/robust-disk-test_2017_08_23_17_06_26_0001/itr_150/params.pkl"

NUM_TIME_STEPS = 500
# NUM_STATES = 30
NUM_STATES = 5

def partition_sampled_states(states, lb, spacing, num_grid = NUM_GRID, transform = lambda x: x[-2:]):
    """
    Given list of states, returns list of lists of states in each grid
    Assumes square big grid
    lb: lower bound
    spacing: space between grid
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
            print("States processed:", i)
        x, y = transform(state)
        x_index = min(int((x - lb) / spacing), num_grid - 1)
        y_index = min(int((y - lb) / spacing), num_grid - 1)
        grid_states[x_index][y_index].append(state)
    return grid_states

def trim_data_set(data, max_states = 5000, lb = -0.05, spacing = 0.02, num_grid = NUM_GRID):
    print("Dividing states into grid spaces")
    grid_states = partition_sampled_states(data, lb, spacing, num_grid)
    trimmed_set = []
    for i in range(num_grid):
        for j in range(num_grid):
            grid = grid_states[i][j]
            num_states_in_grid = len(grid)
            if num_states_in_grid < max_states:
                print("Warning: only {} states in position {} {}".format(num_states_in_grid, i, j))
                trimmed_set.extend(grid)
            else:
                indices = np.random.choice(num_states_in_grid, max_states, replace=False)
                # import ipdb; ipdb.set_trace()
                trimmed_set.extend(np.array(grid)[indices].tolist())
    trimmed_set = np.array(trimmed_set)
    trimmed_starts = StateCollection(distance_threshold=all_feasible_starts.distance_threshold)
    print("Total number of states (before appending to state collection): {}".format(len(trimmed_set)))
    trimmed_starts.append(trimmed_set)
    print("Total number of states (after appending to state collection): {}".format(trimmed_starts.size))

    file = args.file + "trimmed{}.pkl".format(max_states)
    with open(file, "wb") as f:
        pickle.dump(trimmed_starts, f)
    return

def eval_success_grid(data, save_dir="success_breakdown.csv"):
    print("Dividing states into grid spaces")
    grid_states = partition_sampled_states(data, -0.05, 0.02, NUM_GRID)
    data = joblib.load(POLICY_PATH)
    if "algo" in data:
        policy = data["algo"].policy
        env = data["algo"].env
    else:
        policy = data['policy']
        env = data['env']

    with open('data/' + save_dir, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["i", "j", "num_states", "success"])
        for i in range(NUM_GRID):
            for j in range(NUM_GRID):
                plot_num = i * NUM_GRID + j
                print("Progress", plot_num)
                grid = grid_states[i][j]
                num_states_in_grid = len(grid)
                num_success = 0
                folder = "gen_states_visual/data_{}_{}/".format(i, j)

                # may cause race condition
                if not os.path.exists(folder):
                    os.makedirs(folder)

                # sample states for rollout
                indices = np.random.choice(num_states_in_grid, NUM_STATES, replace=False)
                for n, index in enumerate(indices):
                    sampled_state = grid[index]
                    print(np.sum(sampled_state))
                    env.update_start_generator(FixedStateGenerator(sampled_state))
                    env.reset()


                    #unplot for roll out
                    path = rollout(env, policy, 500, animated=False, speedup=2)
                    success = path["rewards"][-1]
                    num_success += success
                    success_rate = num_success * 1.0 / NUM_STATES
                out = i, j, num_states_in_grid, success_rate
                print(out)
                csvwriter.writerow(out)
    return

def grid_and_sample_states(data, save_images = False, file_name ='num_starts_breakdown.csv'):
    # Iterates through points in the grid and performs rollouts to estimate percentage of success

    print("Dividing states into grid spaces")
    grid_states = partition_sampled_states(data, -0.05, 0.02, NUM_GRID)
    save_dir = "data/" # us.uped to save images of state
    data = joblib.load(POLICY_PATH)
    if "algo" in data:
        policy = data["algo"].policy
        env = data["algo"].env
    else:
        policy = data['policy']
        env = data['env']

    # viewer = env.get_viewer()
    cam_pos = [0, 0.6, 0.5, 0.75, -60, 270]


    # don't actually need CSV here ... can remove
    with open('data/' + file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["i", "j", "num_states"])
        # fig, axes = plt.subplots(NUM_GRID * NUM_GRID, NUM_STATES, figsize=(NUM_STATES * 2, NUM_GRID * NUM_GRID * 2))
        # fig, axes = plt.subplots(1, NUM_STATES, figsize=(500, 2500))
        # result = Image.new("RGB", (500, 2500))
        result = np.zeros((NUM_STATES * 100, NUM_GRID * NUM_GRID * 100, 3))
        for i in range(NUM_GRID):
            for j in range(NUM_GRID):
                plot_num = i * NUM_GRID + j # range from 0 to NUM_GRID * NUM_GRID - 1
                print("Progress", plot_num)
                grid = grid_states[i][j]
                num_states_in_grid = len(grid)
                num_success = 0
                folder = "gen_states_visual/data_{}_{}/".format(i, j)

                # may cause race condition
                if not os.path.exists(folder):
                    os.makedirs(folder)


                indices = np.random.choice(num_states_in_grid, NUM_STATES, replace=False)
                for n, index in enumerate(indices):
                    sampled_state = grid[index]
                    print(np.sum(sampled_state))
                    env.update_start_generator(FixedStateGenerator(sampled_state))
                    env.reset()

                    if save_images:
                        viewer = env.get_viewer()
                        env.wrapped_env.setup_camera(cam_pos, viewer)
                        rgb = env.render(mode="rgb_array")
                        # print(rgb.shape)
                        rgb = scipy.misc.imresize(rgb, (100, 100, 3))
                        print(rgb)
                        # print(rgb.shape)
                        scipy.misc.imsave(folder + "{}.jpg".format(n), rgb)
                        # coord = (n * 100, plot_num * 100)
                        # print(coord)

                        result[n * 100: (n + 1) * 100, plot_num * 100: (plot_num + 1) * 100, :] = rgb
                        # result.paste(rgb, coord)
                        # ax = axes[plot_num][n]
                        # ax.imshow(rgb)
                        # ax.axis('off')
                out = i, j, num_states_in_grid
                print(out)
                csvwriter.writerow(out)
        # plt.tight_layout()
        # plt.show()
        scipy.misc.imsave('gen_states.jpg', result)
        # fig.savefig("generated_states.png")

def plot_peg_position_density(data, num_bins = 5, bound =-0.05):
    x_peg = data[:, peg_joints[0]]
    y_peg = data[:, peg_joints[1]]
    fig, ax = plt.subplots()
    heatmap, xedges, yedges = np.histogram2d(x_peg,
                                             y_peg,
                                             range=[[bound, -bound], [bound, -bound]],
                                             bins=[num_bins, num_bins])
    heatmap /= np.sum(heatmap)
    # Plot peg density
    # eventual todo? we can use grid_states to plot density heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    fig, ax = plt.subplots()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap="Blues")
    plt.colorbar()
    plt.title("Generated peg positions density")
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    fig.savefig(args.file + "peg_densities.png")

    # Plots x-y peg positions
    fig, ax = plt.subplots()
    plt.scatter(x_peg, y_peg, c="blue")
    plt.xlim([-0.2, 0.2])
    plt.ylim([-0.2, 0.2])
    plt.title("Generated peg positions")
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
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
    # plt.show()
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
    POLICY_PATH = "data/s3/robust-disk-test5/robust-disk-test5_2017_08_31_11_04_43_0001/itr_60/params.pkl"
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

    # trims data set and saves collection in pkl file!
    trim_data_set(data, max_states=50, lb=-0.03, spacing=0.01, num_grid =6)

    plot_peg_position_density(data, bound=-0.03, num_bins= 6)
    plot_joint_variations(data)
    show_generated_states(data)

    # eval_success_grid(data, "success_breakdown_new_policy.csv")
    # grid_and_sample_states(data, save_images=True, file_name="success_breakdown_num_states.csv")
    #
    plot_peg_position_density(data)
    plot_joint_variations(data)
    # show_generated_states(data)



