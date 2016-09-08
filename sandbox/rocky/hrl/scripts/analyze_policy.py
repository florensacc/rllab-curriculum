


import joblib
import ipdb
from rllab.misc.special import to_onehot
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
import numpy as np


def print_env(env):
    for x in range(env.size):
        for y in range(env.size):
            if (x, y) in env.object_positions:
                idx = env.visit_order.index(env.object_positions.index((x, y)))
                if idx >= env.n_visited:
                    s = str(idx)
                else:
                    s = "X"
            else:
                s = "X"
            if (x, y) == env.agent_pos:
                s = "(" + s + ")"
            else:
                s = " " + s + " "
            print(s, end="")
        print()


if __name__ == "__main__":
    data = joblib.load("/Users/dementrock/Downloads/itr_499(2).pkl")
    policy = data['policy']
    env = data['env']

    assert isinstance(policy, StochasticGRUPolicy)
    assert isinstance(env, PermGridEnv)

    obs = env.reset()
    env.visit_order = tuple(range(env.n_objects))
    env.agent_pos = (2, 2)

    # policy.reset()
    #
    # t = 0
    max_path_length = 100
    # while t < max_path_length:
    #     action, _ = policy.get_action(env.get_current_obs())
    #     _, _, done, _ = env.step(action)
    #     print("T:", t)
    #     print("Hidden state:", policy.hidden_state)
    #     print_env(env)
    #     t += 1
    #     if done:
    #         break

    # print("Map of the grid:")
    # for x in range(env.size):
    #     for y in range(env.size):
    #         if (x, y) in env.object_positions:
    #             env.object_positions.index((x, y))
    print("Fixed visit order to %s" % str(env.visit_order))
    print("Map of the grid:")

    bos = env.get_current_obs()
    n_rollouts = 100

    pos_space = Discrete(env.size ** 2)

    for goal_idx in range(policy.n_subgoals):
        # We can compute the stationary distribution of states following each goal
        records = []
        for _ in range(n_rollouts):
            env.reset()
            env.visit_order = tuple(range(env.n_objects))
            t = 0
            while t < max_path_length:
                xpos, ypos = env.agent_pos
                pos = xpos * env.size + ypos
                records.append(pos_space.flatten(pos))
                policy.hidden_state = to_onehot(goal_idx, policy.n_subgoals)
                action, _ = policy.get_action(env.get_current_obs())
                _, _, done, _ = env.step(action)
                t += 1
                if done:
                    break
        freq = np.mean(records, axis=0)
        freq_grid = freq.reshape((env.size, env.size))
        print("Goal %d" % goal_idx)
        for x in range(env.size):
            print(*('%.3f' % freq_grid[x][y] for y in range(env.size)), sep=' | ')
