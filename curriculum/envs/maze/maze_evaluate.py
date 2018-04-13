import argparse
import os

import uuid
import joblib

from matplotlib import pyplot as plt
import numpy as np
from pylab import *
import pylab
import matplotlib.colorbar as cbar
import matplotlib.patches as patches

from rllab.sampler.utils import rollout
from rllab.misc import logger
from curriculum.envs.base import FixedStateGenerator
# from curriculum.state.selectors import FixedStateSelector
from curriculum.state.evaluator import evaluate_states
from curriculum.logging.visualization import save_image

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


def unwrap_maze(env):
    obj = env
    while not hasattr(obj, 'find_empty_space') and hasattr(obj, 'wrapped_env'):
        obj = obj.wrapped_env
    assert hasattr(obj, 'find_empty_space'), "Your train env has not find_empty_spaces!"
    return obj


def sample_unif_feas(train_env, samples_per_cell):
    """
    :param train_env: wrappers around maze
    :param samples_per_cell: how many samples per cell of the maze
    :return:
    """
    maze_env = unwrap_maze(train_env)
    empty_spaces = maze_env.find_empty_space()

    size_scaling = maze_env.MAZE_SIZE_SCALING

    states = []
    for empty_space in empty_spaces:
        for i in range(samples_per_cell):
            state = np.array(empty_space) + np.random.uniform(-size_scaling/2, size_scaling/2, 2)
            states.append(state)

    return np.array(states)


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


def plot_heatmap(rewards, goals, prefix='', spacing=1, show_heatmap=True, maze_id=0,
                 limit=None, center=None, adaptive_range=False):
    fig, ax = plt.subplots()

    x_goal, y_goal = np.array(goals)[:, :2].T

    if adaptive_range:
        my_square_scatter(axes=ax, x_array=x_goal, y_array=y_goal, z_array=rewards, min_z=np.min(rewards),
                          max_z=np.max(rewards), size=spacing)
    else:
        # THIS IS FOR BINARY REWARD!!!
        my_square_scatter(axes=ax, x_array=x_goal, y_array=y_goal, z_array=rewards, min_z=0, max_z=1, size=spacing)

    if maze_id == 0:
        ax.add_patch(patches.Rectangle((-3, -3), 10, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -3), 2, 10, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, 5), 10, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((5, -3), 2, 10, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-1, 1), 4, 2, fill=True, edgecolor="none", facecolor='0.4'))
    elif maze_id == 11:
        ax.add_patch(patches.Rectangle((-7, 5), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((5, -7), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -7), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -7), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, 1), 10, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -3), 2, 6, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -3), 6, 2, fill=True, edgecolor="none", facecolor='0.4'))
    elif maze_id == 12:
        ax.add_patch(patches.Rectangle((-7, 5), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((5, -7), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -7), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -7), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
    if limit is not None:
        if center is None:
            center = np.zeros(2)
        ax.set_ylim(center[0] - limit, center[0] + limit)
        ax.set_xlim(center[1] - limit, center[1] + limit)

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


def test_policy(policy, train_env, as_goals=True, visualize=True, sampling_res=1, n_traj=1, parallel=True,
                bounds=None, center=None):

    if parallel:
        return test_policy_parallel(policy, train_env, as_goals, visualize, sampling_res, n_traj=n_traj,
                                    center=center, bounds=bounds)

    logger.log("Not using the parallel evaluation of the policy!")
    if hasattr(train_env.wrapped_env, 'find_empty_space'):
        maze_env = train_env.wrapped_env
    else:
        maze_env = train_env.wrapped_env.wrapped_env
    empty_spaces = maze_env.find_empty_space()

    old_goal_generator = train_env.goal_generator if hasattr(train_env, 'goal_generator') else None
    old_start_generator = train_env.start_generator if hasattr(train_env, 'start_generator') else None

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
    avg_time = []
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
                    init_state = np.zeros_like(old_start_generator.state)
                    init_state[:2] = (x, y)
                    states.append(init_state)
                    train_env.update_init_selector(FixedStateGenerator(init_state))
                    print(init_state)
                for n in range(n_traj):
                    path = rollout(train_env, policy, animated=visualize, max_path_length=max_path_length, speedup=100)
                    paths.append(path)
                avg_totRewards.append(np.mean([np.sum(path['rewards']) for path in paths]))
                avg_success.append(np.mean([int(np.min(path['env_infos']['distance'])
                                                <= train_env.terminal_eps) for path in paths]))
                avg_time.append(np.mean([path['rewards'].shape[0] for path in paths]))

    return avg_totRewards, avg_success, states, spacing, avg_time


def find_empty_spaces(train_env, sampling_res=1):
    if hasattr(train_env.wrapped_env, 'find_empty_space'):
        maze_env = train_env.wrapped_env
    else:
        maze_env = train_env.wrapped_env.wrapped_env
    empty_spaces = maze_env.find_empty_space()

    size_scaling = maze_env.MAZE_SIZE_SCALING
    num_samples = 2 ** sampling_res
    spacing = size_scaling / num_samples
    starting_offset = spacing / 2

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
    if quick_test:
        empty_spaces = empty_spaces[:3]

    for empty_space in empty_spaces:
        starting_x = empty_space[0] - size_scaling / 2 + starting_offset
        starting_y = empty_space[1] - size_scaling / 2 + starting_offset
        for i in range(num_samples):
            for j in range(num_samples):
                x = starting_x + i * spacing
                y = starting_y + j * spacing
                states.append((x, y))
    return np.array(states), spacing


def tile_space(bounds, sampling_res=0):
    """sampling_res: how many times split in 2 the axes"""
    assert np.size(bounds[0]) == np.size(bounds[1]), "the bounds are not the same dim!"
    num_samples = 2. ** sampling_res  # num_splits of the axis
    spacing = 1. / num_samples
    starting_offset = spacing / 2

    axes = []
    for idx in range(np.size(bounds[0])):
        axes.append(np.linspace(bounds[0][idx] + starting_offset, bounds[1][idx] - starting_offset,
                                2**sampling_res * (bounds[1][idx] - bounds[0][idx])))
    states = zip(*[g.flat for g in np.meshgrid(*axes)])
    return states, spacing


def test_policy_parallel(policy, train_env, as_goals=True, visualize=True, sampling_res=1, n_traj=1,
                         center=None, bounds=None):
    old_goal_generator = train_env.goal_generator if hasattr(train_env, 'goal_generator') else None
    old_start_generator = train_env.start_generator if hasattr(train_env, 'start_generator') else None
    gen_state_size = np.size(old_goal_generator.state) if old_goal_generator is not None \
                else np.size(old_start_generator)

    if quick_test:
        sampling_res = 0
        max_path_length = 100
    else:
        max_path_length = 400

    if bounds is not None:
        if np.array(bounds).size == 1:
            bounds = [-1 * bounds * np.ones(gen_state_size), bounds * np.ones(gen_state_size)]
        states, spacing = tile_space(bounds, sampling_res)
    else:
        states, spacing = find_empty_spaces(train_env, sampling_res=sampling_res)

    # hack to adjust dim of starts in case of doing velocity also
    states = [np.pad(s, (0, gen_state_size - np.size(s)), 'constant') for s in states]

    avg_totRewards = []
    avg_success = []
    avg_time = []
    logger.log("Evaluating {} states in a grid".format(np.shape(states)[0]))
    rewards, paths = evaluate_states(states, train_env, policy, max_path_length, as_goals=as_goals, n_traj=n_traj, full_path=True)
    logger.log("States evaluated")

    path_index = 0
    for _ in states:
        state_paths = paths[path_index:path_index + n_traj]
        avg_totRewards.append(np.mean([np.sum(path['rewards']) for path in state_paths]))
        avg_success.append(np.mean([int(np.min(path['env_infos']['distance'])
                                        <= train_env.terminal_eps) for path in state_paths]))
        avg_time.append(np.mean([path['rewards'].shape[0] for path in state_paths]))

        path_index += n_traj
    return avg_totRewards, avg_success, states, spacing, avg_time


def test_and_plot_policy(policy, env, as_goals=True, visualize=True, sampling_res=1,
                         n_traj=1, max_reward=1, itr=0, report=None, center=None, limit=None, bounds=None):

    avg_totRewards, avg_success, states, spacing, avg_time = test_policy(policy, env, as_goals, visualize, center=center,
                                                               sampling_res=sampling_res, n_traj=n_traj, bounds=bounds)
    obj = env
    while not hasattr(obj, '_maze_id') and hasattr(obj, 'wrapped_env'):
        obj = obj.wrapped_env
    maze_id = obj._maze_id if hasattr(obj, '_maze_id') else None
    plot_heatmap(avg_success, states, spacing=spacing, show_heatmap=False, maze_id=maze_id,
                 center=center, limit=limit)
    reward_img = save_image()

    # plot_heatmap(avg_time, states, spacing=spacing, show_heatmap=False, maze_id=maze_id,
    #              center=center, limit=limit, adaptive_range=True)
    # time_img = save_image()

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
        # report.add_image(
        #     time_img,
        #     'policy time\n itr: {} \n'.format(
        #         itr
        #     )
        # )
    return mean_rewards, success


def plot_policy_means(policy, env, sampling_res=2, report=None, center=None, limit=None):  # only for start envs!
    states, spacing = find_empty_spaces(env, sampling_res=sampling_res)
    goal = env.current_goal
    observations = [np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state) - len(goal)), goal]) for state in states]
    actions, agent_infos = policy.get_actions(observations)
    vecs = agent_infos['mean']
    vars = [np.exp(log_std) * 0.25 for log_std in agent_infos['log_std']]
    ells = [patches.Ellipse(state, width=vars[i][0], height=vars[i][1], angle=0) for i, state in enumerate(states)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for e in ells:
        ax.add_artist(e)
        e.set_alpha(0.2)
    plt.scatter(*goal, color='r', s=100)
    Q = plt.quiver(states[:,0], states[:,1], vecs[:, 0], vecs[:, 1], units='xy', angles='xy', scale_units='xy', scale=1)  # , np.linalg.norm(vars * 4)
    qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
    # cb = plt.colorbar(Q)
    vec_img = save_image()
    if report is not None:
        report.add_image(vec_img, 'policy mean')


def plot_policy_values(env, baseline, sampling_res=2, report=None, center=None, limit=None):  # TODO: try other baseline
    states, spacing = find_empty_spaces(env, sampling_res=sampling_res)
    goal = env.current_goal
    observations = [np.concatenate([state, [0, 0], goal]) for state in states]
    return




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
