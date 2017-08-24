import random

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

from rllab.algos.nop import NOP
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.utils import rollout
from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.young_clgan.experiments.starts.robust_disk.disk_generate_states_env import DiskGenerateStatesEnv


import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

"""
Inspect generated states
"""


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

x_peg = data[:, peg_joints[0]]
y_peg = data[:, peg_joints[1]]
fig, ax = plt.subplots()
heatmap, xedges, yedges = np.histogram2d(x_peg,
                                         y_peg,
                                         bins=[10, 10])
heatmap /= np.sum(heatmap)
# import pdb; pdb.set_trace()

# Plot peg density
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
