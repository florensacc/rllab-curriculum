import argparse
import csv
import pickle
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.utils import rollout
from sandbox.young_clgan.robust_disk.envs.disk_generate_states_env import DiskGenerateStatesEnv
from sandbox.young_clgan.envs.base import FixedStateGenerator
import sys

"""
Various utils to inspect/evaluate generated states
"""
NUM_GRID = 5
POLICY_PATH = "data/s3/robust-disk-test/robust-disk-test_2017_08_23_09_58_01_0001/itr_160/params.pkl"
NUM_TIME_STEPS = 500
NUM_STATES = 30

def partition_sampled_states(states, lb, spacing, num_grid = NUM_GRID, transform = lambda x: x[-2:]):
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

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str,
                    help='path to the directory with states generated')
args = parser.parse_args()
# args.file = "/home/michael/rllab_goal_rl/data/local/robust-disk-gen-states-uniform2/robust-disk-gen-states-uniform2_2017_08_20_16_02_56_0001/"
file = args.file + "all_feasible_states.pkl"
all_feasible_starts = pickle.load(open(file, "rb"))

num_states = all_feasible_starts.size
print("Getting data from: ", file)
print("Number of states: ", num_states)
data = all_feasible_starts.states
peg_joints = 7, 8

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

# Iterates through points in the grid and performs rollouts to estimate percentage of success
with open('data/success_breakdown.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["i", "j", "num_states", "success"])
    for i in range(NUM_GRID):
        for j in range(NUM_GRID):
            grid = grid_states[i][j]
            num_states_in_grid = len(grid)
            num_success = 0
            indices = np.random.choice(num_states_in_grid, NUM_STATES, replace=False)
            for index in indices:
                sampled_state = grid[index]
                env.update_start_generator(FixedStateGenerator(sampled_state))
                path = rollout(env, policy, 500, False)
                success = path["rewards"][-1]
                num_success += success
            out = i, j, num_states_in_grid, num_success * 1.0 / NUM_STATES
            print(out)
            csvwriter.writerow(out)



sys.exit(0)

x_peg = data[:, peg_joints[0]]
y_peg = data[:, peg_joints[1]]
fig, ax = plt.subplots()
heatmap, xedges, yedges = np.histogram2d(x_peg,
                                         y_peg,
                                         bins=[10, 10])
heatmap /= np.sum(heatmap)
# import pdb; pdb.set_trace()

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

# Simulates policy to visualize generated states
env = DiskGenerateStatesEnv(evaluate = True)
while True:
    policy = GaussianMLPPolicy(
            env_spec=env.spec,
        )
    rollout(env, policy, animated=True, max_path_length=10, init_state=data[int(random.random() * num_states)])


# fig.savefig("joint_variations.png")
# fig.savefig("peg.png")
