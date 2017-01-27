import argparse
import joblib
import uuid
import tensorflow as tf
import numpy as np
from sandbox.haoran.myscripts.myutilities import get_true_env
from gym.envs.mujoco import mujoco_env
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy

from rllab.sampler.utils import rollout
from rllab.misc.ext import set_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--seed', type=int, default=-1,
        help='Fixed random seed for each rollout. Set for reproducibility.')
    parser.add_argument('--no-plot', default=False, action='store_true')
    args = parser.parse_args()

    with tf.Session() as sess:
        data = joblib.load(args.file)
        if "algo" in data:
            algo = data["algo"]
            if hasattr(algo, "eval_policy") and \
                (algo.eval_policy is not None):
                policy = algo.eval_policy
            else:
                policy = algo.policy
            env = data["algo"].env
        else:
            policy = data['policy']
            env = data['env']
        while True:
            if args.seed >= 0:
                set_seed(args.seed)
                true_env = get_true_env(env)
                if isinstance(true_env, mujoco_env.MujocoEnv):
                    # Gym mujoco env doesn't use the default np seed
                    true_env._seed(args.seed)
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=(not args.no_plot), speedup=args.speedup)
