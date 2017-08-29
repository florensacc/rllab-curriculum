import os.path as osp
import argparse
import pickle
import joblib
import tensorflow as tf

from rllab.sampler.utils import rollout
from rllab.misc.ext import set_seed
from sandbox.young_clgan.envs.base import FixedStateGenerator, UniformListStateGenerator
import csv

"""
Evaluates new files on old test set
"""

def eval_old_test_set(env, policy, starts, num_rollouts = 1000, max_path_length = 500):

    # collection_file = "data_upload/state_collections/disc_all_feasible_states_min.pkl"
    # starts = pickle.load(open(collection_file, 'rb'))

    num_success = 0
    for i in range(num_rollouts):
        init_states = starts.sample(10000)
        env.update_start_generator(UniformListStateGenerator(init_states))
        path = rollout(env, policy, max_path_length=max_path_length,
                       animated=False)
        success = path["rewards"][-1]
        num_success += success
        if i > 0 and i % 100 == 0:
            print("Rollouts: {}  Success: {}".format(i, num_success * 1.0 / num_rollouts))
    return num_success


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file') # actually directory
    parser.add_argument('--max_path_length', type=int, default=1000,
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

    with open('data/policy_benchmark.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Iteration", "Success", "Success New Set", "Success New Set Longer Rollout"])

        if args.seed >= 0:
            set_seed(args.seed)
        if args.collection_file:
            all_feasible_starts = pickle.load(open(args.collection_file, 'rb'))

        rollouts = 1000

        # Old set where peg is stationary
        collection_file = "data_upload/state_collections/disc_all_feasible_states_min.pkl"
        all_feasible_starts = pickle.load(open(collection_file, 'rb'))

        # Test set is trained
        collection_file2 = "data_upload/peg/euclidian_joint_distance/all_feasible_states.pkl"
        all_feasible_starts2 = pickle.load(open(collection_file, 'rb'))

        with tf.Session() as sess:
            for i in range(17):
                print("Testing on iteration: ", i)
                iteration = i * 10
                file = args.file + "itr_{}/params.pkl".format(i * 10)
                data = joblib.load(file)
                if "algo" in data:
                    policy = data["algo"].policy
                    env = data["algo"].env
                else:
                    policy = data['policy']
                    env = data['env']

                print("Old Test Set")
                success_old_set = eval_old_test_set(env, policy, all_feasible_starts, num_rollouts=rollouts)

                print("New Test Set")
                success_robust_set = eval_old_test_set(env, policy, all_feasible_starts2, num_rollouts=rollouts)

                print("New Test Set Longer Rollout")
                success_robust_set_long_rollout = eval_old_test_set(env, policy, all_feasible_starts2, num_rollouts=rollouts,
                                                       max_path_length = 1000)


                csvwriter.writerow([iteration, success_old_set * 1.0 / rollouts, success_robust_set * 1.0 / rollouts,
                                    success_robust_set_long_rollout * 1.0 / rollouts])



