import multiprocessing
import os
import tempfile
import numpy as np
from collections import OrderedDict
import cloudpickle
import time

from rllab.sampler.utils import rollout
from rllab.misc import logger

from sandbox.young_clgan.envs.base import FixedStateGenerator

class FunctionWrapper(object):
    """Wrap a function for use with parallelized map.
    """

    def __init__(self, func, *args, **kwargs):
        """Construct the function oject.
        Args:
          func: a top level function, or a picklable callable object.
          *args and **kwargs: Any additional required enviroment data.
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, obj):
        if obj is None:
            return self.func(*self.args, **self.kwargs)
        else:
            return self.func(obj, *self.args, **self.kwargs)

    def __getstate__(self):
        """ Here we overwrite the default pickle protocol to use cloudpickle. """
        return dict(
            func=cloudpickle.dumps(self.func),
            args=cloudpickle.dumps(self.args),
            kwargs=cloudpickle.dumps(self.kwargs)
        )

    def __setstate__(self, d):
        self.func = cloudpickle.loads(d['func'])
        self.args = cloudpickle.loads(d['args'])
        self.kwargs = cloudpickle.loads(d['kwargs'])


def disable_cuda_initializer(*args, **kwargs):
    import os
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def parallel_map(func, iterable_object, num_processes=-1):
    """Parallelized map function based on python process
    Args:
    func: Pickleable callable object that takes one parameter.
    iterable_object: An iterable of elements to map the function on.
    num_processes: Number of process to use. When num_processes is 1,
                   no new process will be created.
    Returns:
    The list resulted in calling the func on all objects in the original list.
    """
    if num_processes == 1:
        return [func(x) for x in iterable_object]
    if num_processes == -1:
        from rllab.sampler.stateful_pool import singleton_pool
        num_processes = singleton_pool.n_parallel
    process_pool = multiprocessing.Pool(
        num_processes,
        initializer=disable_cuda_initializer
    )
    results = process_pool.map(func, iterable_object)
    process_pool.close()
    process_pool.join()
    return results

def compute_rewards_from_paths(all_paths, key='rewards', as_goal=True, env=None):
    all_rewards = []
    all_states = []
    for paths in all_paths:
        for path in paths:
            reward = evaluate_path(path, key=key)
            if as_goal:
                state = tuple(path['env_infos']['goal'][0])
            else:
                state = tuple(env.transform_to_start_space(path['observations'][0]))

            all_states.append(state)
            all_rewards.append(reward)

    return [all_states, all_rewards]


def label_states_from_paths(all_paths, min_reward=0, max_reward=1, key='rewards', as_goal=True,
                 old_rewards=None, improvement_threshold=0, n_traj=1, env=None):
    state_dict = {}
    for paths in all_paths:
        for path in paths:
            reward = evaluate_path(path, key=key)
            if as_goal:
                state = tuple(path['env_infos']['goal'][0])
            else:
                state = tuple(env.transform_to_start_space(path['observations'][0]))
            if state in state_dict:
                state_dict[state].append(reward)
            else:
                state_dict[state] = [reward]

    states = []
    mean_rewards = []
    for state, rewards in state_dict.items():
        if len(rewards) >= n_traj:
            states.append(list(state))
            mean_rewards.append(np.mean(rewards))

    # Make this a vertical list.
    mean_rewards = np.array(mean_rewards).reshape(-1, 1)

    labels = compute_labels(mean_rewards, old_rewards=old_rewards, min_reward=min_reward, max_reward=max_reward,
                            improvement_threshold=improvement_threshold)

    states = np.array(states)

    return [states, labels]


def label_states(states, env, policy, horizon, as_goals=True, min_reward=0.1, max_reward=0.9, key='rewards',
                 old_rewards=None, improvement_threshold=0, n_traj=1, n_processes=-1, full_path=False):
    logger.log("Labelling starts")
    result = evaluate_states(
        states, env, policy, horizon, as_goals=as_goals,
        n_traj=n_traj, n_processes=n_processes, key=key, full_path=full_path
    )
    if full_path:
        mean_rewards, paths = result
    else:
        mean_rewards = result
    logger.log("Evaluated states.")

    mean_rewards = mean_rewards.reshape(-1, 1)
    labels = compute_labels(mean_rewards, old_rewards=old_rewards, min_reward=min_reward, max_reward=max_reward,
                          improvement_threshold=improvement_threshold)
    logger.log("Starts labelled")

    if full_path:
        return labels, paths
    return labels


def compute_labels(mean_rewards, old_rewards=None, min_reward=0, max_reward=1, improvement_threshold=0):
    logger.log("Computing state labels")
    if old_rewards is not None:
        old_rewards = old_rewards.reshape(-1, 1)
        labels = np.hstack(
            [mean_rewards > min_reward,  # np.zeros_like(mean_rewards > min_reward, dtype=float), #
             mean_rewards < max_reward,  # np.zeros_like(mean_rewards < max_reward, dtype=float),  #
             mean_rewards - old_rewards >= improvement_threshold
             # np.zeros_like(mean_rewards - old_rewards >= improvement_threshold, dtype=float),
             #
             ]
        ).astype(np.float32)
    else:
        labels = np.hstack(
            [mean_rewards > min_reward, mean_rewards < max_reward]
        ).astype(np.float32)

    return labels


def convert_label(labels):
    """
    :param labels: 3-dim evaluation of the state if they have learnability, 2-dim otherwise
    :return: convert to single integer label and gives associated texts (for plotting). Better if OrderedDict for log!
    """
    # label[0] --> LowRew, label[1] --> HighRew, label[2] --> Learnable ??
    # Put good states last so they will be plotted on top of other states and be most visible.
    classes = OrderedDict({
        # 0: r'Low rewards: $\bar{R}<R_{\min}$',
        0: 'Low rewards',
        # 1: r'High rewards: $\bar{R}>R_{\max}$',
        1: 'High rewards',
        2: 'Good goals',
        3: 'Unlearnable',
        4: 'Other',
    })
    new_labels = 4 * np.ones(labels.shape[0], dtype=int)
    new_labels[np.logical_and(labels[:, 0], labels[:, 1])] = 2
    new_labels[labels[:, 0] == False] = 0
    new_labels[labels[:, 1] == False] = 1
    if np.shape(labels)[-1] == 3:
        new_labels[
            np.logical_and(
                np.logical_and(labels[:, 0], labels[:, 1]),
                labels[:, 2] == False
            )
        ] = 3

    return new_labels, classes


def evaluate_states(states, env, policy, horizon, n_traj=1, n_processes=-1, full_path=False, key='rewards',
                    as_goals=True,
                    aggregator=(np.sum, np.mean)):
    evaluate_state_wrapper = FunctionWrapper(
        evaluate_state,
        env=env,
        policy=policy,
        horizon=horizon,
        n_traj=n_traj,
        full_path=full_path,
        key=key,
        as_goals=as_goals,
        aggregator=aggregator,
    )
    result = parallel_map(  # if full_path this is a list of tuples
        evaluate_state_wrapper,
        states,
        n_processes,
    )

    if full_path:
        return np.array([state[0] for state in result]), [path for state in result for path in state[1]]
    return np.array(result)


def evaluate_state(state, env, policy, horizon, n_traj=1, full_path=False, key='rewards', as_goals=True,
                   aggregator=(np.sum, np.mean)):
    aggregated_data = []
    paths = []
    if as_goals:
        env.update_goal_generator(FixedStateGenerator(state))
    else:
        env.update_start_generator(FixedStateGenerator(state))

    for j in range(n_traj):
        paths.append(rollout(env, policy, horizon))

        if key in paths[-1]:
            aggregated_data.append(
                aggregator[0](paths[-1][key])
            )
        else:
            aggregated_data.append(
                aggregator[0](paths[-1]['env_infos'][key])
            )

    mean_reward = aggregator[1](aggregated_data)

    if full_path:
        return mean_reward, paths

    return mean_reward

def evaluate_state_env(env, policy, horizon, n_states=10, n_traj=1, n_processes=-1, **kwargs):
    evaluate_env_wrapper = FunctionWrapper(
        rollout,
        env=env, agent=policy, max_path_length=horizon,
    )
    paths = parallel_map(evaluate_env_wrapper, [None] * n_states, n_processes)

    # paths = [rollout(env=env, agent=policy, max_path_length=horizon) for _ in range(n_states)]
    env.log_diagnostics(paths, n_traj=n_traj, **kwargs)

def evaluate_path(path, full_path=False, key='rewards', aggregator=np.sum):
    if not full_path:
        if key in path:
            total_reward = aggregator(path[key])
        else:
            total_reward = aggregator(path['env_infos'][key])
        return total_reward

    if full_path:
        return path


