# Example run
## python3 sandbox/tuomas/mddpg/scripts/eval_rough_terrain.py data/s3/tuomas/hopper/exp-000-ddpg/ --keys "'ou_sigma'" --values "0.03" --speedup 10 --itr 1999 --n_eval=1 --roughnesses='(0.5, 0.6, 0.7, 0.8, 0.9. 1.0)'

import argparse
import joblib
import uuid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sandbox.haoran.myscripts.myutilities import get_true_env
from gym.envs.mujoco import mujoco_env
from rllab.envs.proxy_env import ProxyEnv
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy
from rllab.misc.ext import set_seed
from rllab.misc import tensor_utils
import time

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.haoran.myscripts.envs import EnvChooser
from rllab.envs.normalized_env import normalize

import os
import ast
import json
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('exp_path', type=str, default='xxxxxxxxxxxxxxxxxxxxx',
                    nargs='?')
parser.add_argument('-y', default=False, action='store_true', help='force yes')
parser.add_argument('--keys', type=str, default='()', help='keys')
parser.add_argument('--values', type=str, default='()', help='values')
parser.add_argument('--itr', type=int, default=0, help='epoch number')
parser.add_argument('--speedup', type=int, default=1, help='Speed up')
parser.add_argument('--env', type=str, default='tuomas_hopper',
                    help='Environment name')
parser.add_argument('--roughness', type=str,
                    default='[0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]',
                    help='Scalar or list of roughness levels (0.0 ... 1.0)')
parser.add_argument('--n_eval', type=int, default=1,
                    help='Number of evaluations per policy')
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no-render', dest='render', action='store_false')
parser.set_defaults(render=True)

args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.ERROR)


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0

    true_env = env
    while isinstance(true_env, ProxyEnv):
        true_env = true_env._wrapped_env

    if animated:
        env.render()

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)

        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break

        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

    if animated:
        env.render(close=True)

    return true_env.get_body_com("torso")[0]


def eval_policy(policy, roughness):
    distance_lst = list()
    for i in range(args.n_eval):
        generate_terrain(roughness)

        env_chooser = EnvChooser()
        env = TfEnv(normalize(
            env_chooser.choose_env(args.env, rough_terrain=True),
        ))

        distance_lst.append(
            rollout(env,
                    policy,
                    max_path_length=1000,
                    animated=args.render,
                    speedup=args.speedup)
        )

    return distance_lst


def generate_terrain(roughness_level):
    x = np.abs(np.random.randn(256, 256))
    x = roughness_level * x / np.max(x) * 255
    x[125:131, 125:131] = 0  # Set init location flat.
    # Make sure we always us the full range. Otherwise Mujoco will scale.
    x[0, 0] = 255
    x = x.astype(np.uint8)
    img = Image.fromarray(x)

    file_path = os.path.join(os.path.dirname(__file__),
                             '..', '..', 'assets', 'rough_terrain.png')
    img.save(file_path)


if __name__ == "__main__":
    exp_path = args.exp_path
    runs = os.listdir(exp_path)

    keys = ast.literal_eval(str(args.keys))
    values = ast.literal_eval(str(args.values))

    keys = keys if isinstance(keys, tuple) else (keys,)
    values = values if isinstance(values, tuple) else (values,)

    roughness_lst = ast.literal_eval(args.roughness)
    if isinstance(roughness_lst, float):
        roughness_lst = [roughness_lst]

    # Look for all policies for given criteria.
    policy_file_lst = list()
    for run in runs:
        variant_file = os.path.join(exp_path, run, 'variant.json')

        with open(variant_file) as file:
            variant = json.load(file)

            # Hacky fix: flatten all embedded dictionaries into the root dict.
            embedded = dict()
            for v in variant.values():
                if isinstance(v, dict):
                    embedded.update(v)
            variant.update(embedded)

            if all((variant[k] == v for k, v in zip(keys, values))):

                pkl_file = os.path.join(exp_path, run,
                                        'itr_' + str(args.itr) + '.pkl')
                policy_file_lst.append(pkl_file)

    # Evaluate each roughness level.
    dist_lst_lst = list()
    for i, roughness in enumerate(roughness_lst):
        dist_lst_lst.append(list())
        for j, policy_file in enumerate(policy_file_lst):
            with tf.Session() as sess:
                data = joblib.load(policy_file)
                policy = data['algo'].policy
                dist_lst_lst[-1] += eval_policy(policy, roughness)
            tf.reset_default_graph()
            #print('Policy: ' + str(j))
        print('Roughness level: ' + str(roughness))

    # Compute stats.
    # for each roughness level
    print('roughness iters mean max min var 5th-percentile 95th-percentile')
    for roughness, dist_lst in zip(roughness_lst, dist_lst_lst):
        dists = np.array(dist_lst)
        print(roughness, dists.size, dists.mean(), dists.max(), dists.min(),
              dists.var(), np.percentile(dists, 5), np.percentile(dists, 95))
