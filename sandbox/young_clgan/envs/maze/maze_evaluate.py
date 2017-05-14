import argparse
import os

import uuid
import joblib

from matplotlib import pyplot as plt
import numpy as np
from pylab import *
import pylab
import matplotlib.colorbar as cbar

from rllab.sampler.utils import rollout
from rllab.misc import logger
from sandbox.young_clgan.envs.base import FixedStateGenerator
# from sandbox.young_clgan.state.selectors import FixedStateSelector
from sandbox.young_clgan.state.evaluator import evaluate_states
from sandbox.young_clgan.logging.visualization import save_image

quick_test = False

filename = str(uuid.uuid4())


def get_policy(file):
    policy = None
    train_env = None
    if ':' in file:
        # fetch file using ssh
        os.system("rsync -avrz %s /tmp/%s.pkl" % (file, filename))
        data = joblib.load("/tmp/%s.pkl" % filename)
        if policy:
            new_policy = data['policy']
            policy.set_param_values(new_policy.get_param_values())
        else:
            policy = data['policy']
            train_env = data['env']
    else:
        data = joblib.load(file)
        policy = data['policy']
        train_env = data['env']
    return policy, train_env


def my_square_scatter(axes, x_array, y_array, z_array, min_z=None, max_z=None, size=0.5, **kwargs):
    size = float(size)

    if min_z is None:
        min_z = z_array.min()
    if max_z is None:
        max_z = z_array.max()

    normal = pylab.Normalize(min_z, max_z)
    colors = pylab.cm.jet(normal(z_array))

    for x, y, c in zip(x_array, y_array, colors):
        square = pylab.Rectangle((x - size / 2, y - size / 2), size, size, color=c, **kwargs)
        axes.add_patch(square)

    axes.autoscale()

    cax, _ = cbar.make_axes(axes)
    cb2 = cbar.ColorbarBase(cax, cmap=pylab.cm.jet, norm=normal)

    return True


def plot_heatmap(rewards, goals, prefix='', max_reward=6000, spacing=1, show_heatmap=True):
    fig, ax = plt.subplots()

    x_goal, y_goal = np.array(goals).T

    my_square_scatter(axes=ax, x_array=x_goal, y_array=y_goal, z_array=rewards, min_z=0, max_z=max_reward, size=spacing)

    # colmap = cm.ScalarMappable(cmap=cm.rainbow)
    # colmap.set_array(rewards)
    # Create the contour plot
    # CS = ax.contourf(xs, ys, zs, cmap=plt.cm.rainbow,
    #                   vmax=zmax, vmin=zmin, interpolation='nearest')
    # CS = ax.imshow([rewards], interpolation='none', cmap=plt.cm.rainbow,
    #                vmax=np.max(rewards), vmin=np.min(rewards)) # extent=[np.min(ys), np.max(ys), np.min(xs), np.max(xs)]
    # fig.colorbar(colmap)

    # ax.set_title(prefix + 'Returns')
    # ax.set_xlabel('goal position (x)')
    # ax.set_ylabel('goal position (y)')

    # ax.set_xlim([np.max(ys), np.min(ys)])
    # ax.set_ylim([np.min(xs), np.max(xs)])
    # plt.scatter(x_goal, y_goal, c=rewards, s=1000, vmin=0, vmax=max_reward)
    # plt.colorbar()
    if show_heatmap:
        plt.show()
    return fig


def test_policy(policy, train_env, as_goals=True, visualize=True, sampling_res=1, n_traj=1, parallel=True):
    
    if parallel:
        return test_policy_parallel(policy, train_env, as_goals, visualize, sampling_res, n_traj=n_traj)
    
    
    if hasattr(train_env.wrapped_env, 'find_empty_space'):
        maze_env = train_env.wrapped_env
    else:
        maze_env = train_env.wrapped_env.wrapped_env
    empty_spaces = maze_env.find_empty_space()

    if quick_test:
        sampling_res = 0
        empty_spaces = empty_spaces[:3]
        max_path_length = 100
    else:
        max_path_length = 400

    size_scaling = maze_env.MAZE_SIZE_SCALING
    num_samples = 2 ** sampling_res
    spacing = size_scaling / num_samples
    starting_offset = spacing / 2

    avg_totRewards = []
    avg_success = []
    states = []

    distances = []
    for empty_space in empty_spaces:
        delta_x = empty_space[0]  # - train_env.wrapped_env._init_torso_x
        delta_y = empty_space[1]  # - train_env.wrapped_env._init_torso_y
        distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
        distances.append(distance)

    sort_indices = np.argsort(distances)[::-1]

    empty_spaces = np.array(empty_spaces)
    empty_spaces = empty_spaces[sort_indices]

    for empty_space in empty_spaces:
        starting_x = empty_space[0] - size_scaling / 2 + starting_offset
        starting_y = empty_space[1] - size_scaling / 2 + starting_offset
        for i in range(num_samples):
            for j in range(num_samples):
                paths = []
                x = starting_x + i * spacing
                y = starting_y + j * spacing
                if as_goals:
                    goal = (x, y)
                    states.append(goal)
                    train_env.update_goal_selector(FixedStateGenerator(goal))
                else:
                    init_state = (x, y)
                    states.append(init_state)
                    train_env.update_init_selector(FixedStateGenerator(init_state))
                    # print("fixing init selector to: ", init_state)
                    # print("the goal is set to: ", train_env.current_goal)
                for n in range(n_traj):
                    path = rollout(train_env, policy, animated=visualize, max_path_length=max_path_length, speedup=100)
                    paths.append(path)
                    # print("the first obs in the path is ", path['observations'][0])
                    # print("total length: ", np.size(path['rewards']))
                    # print("the goal is: ", train_env.current_goal)
                    # print("the min dist is: ", np.min(path['env_infos']['distance']))
                # print('goal: ', goal, ', the one in env_infos is: ', paths[-1]['env_infos']['x_goal'], paths[-1]['env_infos']['y_goal'])
                avg_totRewards.append(np.mean([np.sum(path['rewards']) for path in paths]))
                avg_success.append(np.mean([int(np.min(path['env_infos']['distance'])
                                                <= train_env.terminal_eps) for path in paths]))

    return avg_totRewards, avg_success, states, spacing
    
    
def test_policy_parallel(policy, train_env, as_goals=True, visualize=True, sampling_res=1, n_traj=1):
    if hasattr(train_env.wrapped_env, 'find_empty_space'):
        maze_env = train_env.wrapped_env
    else:
        maze_env = train_env.wrapped_env.wrapped_env
    empty_spaces = maze_env.find_empty_space()

    if quick_test:
        sampling_res = 0
        empty_spaces = empty_spaces[:3]
        max_path_length = 100
    else:
        max_path_length = 400

    size_scaling = maze_env.MAZE_SIZE_SCALING
    num_samples = 2 ** sampling_res
    spacing = size_scaling / num_samples
    starting_offset = spacing / 2

    avg_totRewards = []
    avg_success = []
    states = []

    distances = []
    for empty_space in empty_spaces:
        delta_x = empty_space[0]  # - train_env.wrapped_env._init_torso_x
        delta_y = empty_space[1]  # - train_env.wrapped_env._init_torso_y
        distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
        distances.append(distance)

    sort_indices = np.argsort(distances)[::-1]

    empty_spaces = np.array(empty_spaces)
    empty_spaces = empty_spaces[sort_indices]
    
    for empty_space in empty_spaces:
        starting_x = empty_space[0] - size_scaling / 2 + starting_offset
        starting_y = empty_space[1] - size_scaling / 2 + starting_offset
        for i in range(num_samples):
            for j in range(num_samples):
                x = starting_x + i * spacing
                y = starting_y + j * spacing
                states.append((x, y))
    # import pdb; pdb.set_trace()
    paths = evaluate_states(states, train_env, policy, max_path_length, n_traj=n_traj, full_path=True)
   
    path_index = 0
    for empty_space in empty_spaces:
        for i in range(num_samples):
            for j in range(num_samples):
                state_paths = paths[path_index:path_index + n_traj]
                avg_totRewards.append(np.mean([np.sum(path['rewards']) for path in state_paths]))
                avg_success.append(np.mean([int(np.min(path['env_infos']['distance'])
                                                <= train_env.terminal_eps) for path in state_paths]))
                
                path_index += n_traj

    return avg_totRewards, avg_success, states, spacing



def test_and_plot_policy(policy, env, as_goals=True, visualize=True, sampling_res=1,
                         n_traj=1, max_reward=1, itr=0, report=None):
    avg_totRewards, avg_success, states, spacing = test_policy(policy, env, as_goals, visualize,
                                                               sampling_res=sampling_res, n_traj=n_traj)
    plot_heatmap(avg_success, states, max_reward=max_reward, spacing=spacing, show_heatmap=False)

    reward_img = save_image()
    mean_rewards = np.mean(avg_totRewards)
    success = np.mean(avg_success)

    with logger.tabular_prefix('Outer_'):
        logger.record_tabular('iter', itr)
        logger.record_tabular('MeanRewards', mean_rewards)
        logger.record_tabular('Success', success)
    # logger.dump_tabular(with_prefix=False)

    if report is not None:
        report.add_image(
            reward_img,
            'policy performance\n itr: {} \nmean_rewards: {} \nsuccess: {}'.format(
                itr, mean_rewards, success
            )
        )
    return mean_rewards, success





def main():
    # pkl_file = "sandbox/young_clgan/experiments/point_maze/experiment_data/cl_gan_maze/2017-02-20_22-43-48_dav2/log/itr_129/itr_9.pkl"
    #    pkl_file = "sandbox/young_clgan/experiments/point_maze/experiment_data/cl_gan_maze/2017-02-21_15-30-36_dav2/log/itr_69/itr_4.pkl"
    #    pkl_file = "sandbox/young_clgan/experiments/point_maze/experiment_data/cl_gan_maze/2017-02-21_22-49-03_dav2/log/itr_199/itr_4.pkl"
    # pkl_file = "sandbox/young_clgan/experiments/point_maze/experiment_data/cl_gan_maze/2017-02-22_13-06-53_dav2/log/itr_119/itr_4.pkl"
    # pkl_file = "data/local/goalGAN-maze30/goalGAN-maze30_2017_02_24_01_44_03_0001/itr_27/itr_4.pkl"
    pkl_file = "/home/davheld/repos/goalgen/rllab_goal_rl/data/s3/goalGAN-maze11/goalGAN-maze11_2017_02_23_01_06_12_0005/itr_199/itr_4.pkl"

    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--file', type=str, default=pkl_file,
    # #                     help='path to the snapshot file')
    # parser.add_argument('--max_length', type=int, default=100,
    #                     help='Max length of rollout')
    # parser.add_argument('--speedup', type=int, default=1,
    #                     help='Speedup')
    # parser.add_argument('--num_goals', type=int, default=200, #1 * np.int(np.square(0.3/0.02))
    #                     help='Number of test goals')
    # parser.add_argument('--num_tests', type=int, default=1,
    #                     help='Number of test goals')
    # args = parser.parse_args()
    #
    # paths = []

    policy, train_env = get_policy(pkl_file)

    avg_totRewards, avg_success, goals, spacing = test_policy(policy, train_env, sampling_res=1)

    plot_heatmap(avg_totRewards, goals, spacing=spacing)


if __name__ == "__main__":
    main()
