"""
This script helps to conveniently plot the environment and Q values for
multi-headed policies.
"""

from rllab.sampler.utils import rollout
from rllab.misc import tensor_utils
from rllab.envs.proxy_env import ProxyEnv
import argparse
import joblib
import uuid
import os
import random
import numpy as np
import tensorflow as tf
import time
import multiprocessing as mp
from rllab.misc.ext import set_seed


def rollout(process_id, sess,env, agent, exploration_strategy, qf, random=False,
    pause=False, max_path_length=np.inf, animated=False, speedup=1,
    optimal=False, head=-1, window_config=None):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    exploration_strategy.reset()
    if head == -1:
        if exploration_strategy.switch_type == "per_path":
            agent.k = np.mod(agent.k + 1, agent.K)
    else:
        assert head in range(agent.K), "The policy only has %d heads"%(agent.K)
        agent.k = head
    path_length = 0

    if animated:
        env.render(config=window_config)

    while path_length < max_path_length:
        if random:
            a = exploration_strategy.get_action(0, o, policy)
        else:
            if exploration_strategy.switch_type == "per_action" and head != -1:
                agent.k = np.random.randint(low=0, high=agent.K,size=1)
            a, agent_info = agent.get_action(o)
        # print("Process %d"%(process_id), "head: %d"%(agent.k), "action: ", a)

        agent_info = {}
        next_o, r, d, env_info = env.step(a)

        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        if animated:
            env.render(config=window_config)
            timestep = 0.05
            time.sleep(timestep / speedup)
            if pause:
                input()
        o = next_o
    if animated:
        env.render(close=False,config=window_config) # close=True causes the mujoco sim to fail

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

def run(process_id, window_config, seed, args):
    set_seed(seed)
    while True:
        with tf.Session() as sess:
            data = joblib.load(args.file)
            if "algo" in data:
                algo = data["algo"]
                policy = algo.policy
                env = algo.env
                es = algo.exploration_strategy
                qf = algo.qf
            else:
                policy = data['policy']
                env = data['env']
                es = data['es']
                qf = data['qf']
            while True:
                try:
                    path = rollout(
                        process_id,
                        sess,
                        env,
                        policy,
                        es,
                        qf,
                        pause=args.pause,
                        optimal=args.optimal,
                        max_path_length=args.max_path_length,
                        animated=True,
                        speedup=args.speedup,
                        random=args.random,
                        head=args.head,
                        window_config=window_config,
                    )



                # Hack for now. Not sure why rollout assumes that close is an
                # keyword argument
                except TypeError as e:
                    if (str(e) != "render() got an unexpected keyword "
                                  "argument 'close'"):
                        raise e

def organize_windows(
    n_window, n_row, n_col,
    width, height,
    start_xpos, start_ypos,
):
    assert n_row * n_col >= n_window
    xpos_list, ypos_list = [], []
    for i in range(n_row):
        for j in range(n_col):
            xpos = start_xpos + j * width
            ypos = start_ypos + i * height
            xpos_list.append(xpos)
            ypos_list.append(ypos)
    return xpos_list, ypos_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--random', default=False,
        action='store_true')
    parser.add_argument('--pause', default=False,
        action='store_true')
    parser.add_argument('--optimal', default=False,
        action='store_true', help='Use argmax_a Q(s,a) as action.')
    parser.add_argument('--head', default=-1, type=int,
        help='Fix a head throughout the rollout.')
    parser.add_argument('--n-parallel', default=1, type=int,
        help='Number of parallel simulations')
    parser.add_argument('--nrow', default=-1, type=int)
    parser.add_argument('--ncol', default=-1, type=int)
    parser.add_argument('--width', default=400, type=int)
    parser.add_argument('--height', default=400, type=int)
    args = parser.parse_args()


    # choose the right window arrangement
    if args.nrow == -1 and args.ncol == -1:
        n_row = np.floor(np.sqrt(args.n_parallel)).astype(int)
        n_col = np.ceil(args.n_parallel / n_row).astype(int)
    else:
        n_row = args.nrow
        n_col = args.ncol
    width = args.width
    height = args.height

    margin = 5
    xpos_list, ypos_list = organize_windows(
        n_window=args.n_parallel,
        n_row=n_row,
        n_col=n_col,
        width=width + margin,
        height=height + margin,
        start_xpos=0,
        start_ypos=0,
    )

    processes = []
    for i in range(args.n_parallel):
        window_config = dict(
            xpos=xpos_list[i],
            ypos=ypos_list[i],
            width=width,
            height=height,
            title="%d"%(i),
        )
        seed = i
        p = mp.Process(
            target=run,
            args=(i, window_config, seed, args),
        )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()
