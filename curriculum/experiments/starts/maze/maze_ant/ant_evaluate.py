import os.path as osp
import argparse
import pickle
import joblib
import tensorflow as tf

from rllab.sampler.utils import rollout
from rllab.misc.ext import set_seed
from curriculum.envs.base import FixedStateGenerator

def ant_evaluate(env, policy, init_state = None, max_path_length = 2000, animated = True, speedup = 2):
    if init_state is not None:
        if len(init_state) == 2:
            init_state.extend([0.55, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1,]) # first two positions are COM
        env.update_start_generator(FixedStateGenerator(init_state))
    path = rollout(env, policy, max_path_length=max_path_length,
                   animated=animated, speedup=speedup)
    print("Trajectory length: {}".format(len(path["rewards"])))
    print("Success: {}".format(path["rewards"][-1]))
    return path["rewards"][-1]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=2000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Fixed random seed')
    parser.add_argument("-is", '--init_state', type=str,
                        help='vector of init_state')
    parser.add_argument("-cf", '--collection_file', type=str,
                        help='path to the pkl file with start positions Collection')
    args = parser.parse_args()

    policy = None
    env = None

    if args.seed >= 0:
        set_seed(args.seed)
    if args.collection_file:
        all_feasible_starts = pickle.load(open(args.collection_file, 'rb'))

    with tf.Session() as sess:
        data = joblib.load(args.file)
        if "algo" in data:
            policy = data["algo"].policy
            env = data["algo"].env
        else:
            policy = data['policy']
            env = data['env']

        # easiest to hardest
        init_pos = [[0,0]]
        # init_pos = [[0, 0],
        #             [1,0],
        #              [2,0],
        #              [3,0],
        #              [4,0],
        #              [4,1],
        #              [4,2],
        #              [4,3],
        #             [4,4],
        #             [3,4],
        #             [2,4],
        #             [1,4]
        #             ][::-1]
        while True:
            for pos in init_pos:
                path = ant_evaluate(env, policy, init_state= pos,
                            max_path_length=args.max_path_length,
                       animated=True, speedup=args.speedup)

