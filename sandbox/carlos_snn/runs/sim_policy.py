from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import os
import random
import numpy as np

from sandbox.young_clgan.envs.base import FixedStateGenerator

filename = str(uuid.uuid4())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--loop', type=int, default=1,
                        help='# of loops')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed')
    parser.add_argument('--goal', type=str, default='',
                        help='goal state, blank space separated')
    args = parser.parse_args()

    policy = None
    env = None
    while True:
        if ':' in args.file:
            # fetch file using ssh
            os.system("rsync -avrz %s /tmp/%s.pkl" % (args.file, filename))
            data = joblib.load("/tmp/%s.pkl" % filename)
            if parser.seed is not None:
                random.seed(parser.seed)
                np.random.seed(parser.seed)
            if policy:
                new_policy = data['policy']
                policy.set_param_values(new_policy.get_param_values())
                if args.goal:
                    policy.set_pre_fix_latent([int(i) for i in list(args.latent)])
                path = rollout(env, policy, max_path_length=args.max_length,
                               animated=True, speedup=args.speedup)
            else:
                policy = data['policy']
                env = data['env']
                if args.latent:
                    policy.set_pre_fix_latent([int(i) for i in list(args.latent)])
                path = rollout(env, policy, max_path_length=args.max_length,
                               animated=True, speedup=args.speedup)
        else:
            data = joblib.load(args.file)
            policy = data['policy']
            env = data['env']
            if args.goal:
                print(args.goal.split(' '))
                goal = [float(i) for i in args.goal.split(' ')]
                env.update_goal_generator(FixedStateGenerator(goal))
            path = rollout(env, policy, max_path_length=args.max_length,
                           animated=True, speedup=args.speedup)
        # print 'Total reward: ', sum(path["rewards"])
        args.loop -= 1
        if ':' not in args.file:
            if args.loop <= 0:
                break
