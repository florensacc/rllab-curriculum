import argparse

import datetime
import os
import uuid

import dateutil
import joblib

import rllab.misc.logger as logger
from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from rllab import config

from sandbox.dave.pr2_real.pr2_env_lego_real import Pr2EnvReal
from rllab.envs.normalized_env import normalize
import os.path as osp

from sandbox.dave.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2CrownGoalGeneratorSmall  #PR2FixedGoalGenerator #PR2CrownGoalGeneratorSmall #PR2TestGoalGenerator

from rllab.sampler.utils import rollout

filename = str(uuid.uuid4())


def do_test(env, policy, num_test_goals, max_path_length):
    for itr in range(num_test_goals):
        with logger.prefix('itr #%d | ' % itr):
            path = rollout(env, policy, animated=True, max_path_length=max_path_length, speedup=10)
            paths = [path]
            env.log_diagnostics(paths)
            policy.log_diagnostics(paths)

            #logger.dump_tabular(with_prefix=True)
            # if self.plot:
            #     self.update_plot()
            #     if self.pause_for_plot:
            #         raw_input("Plotting evaluation run: Press Enter to "
            #                   "continue...")


def setup_logging():
    #default_log_dir = config.LOG_DIR
    #now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    #rand_id = str(uuid.uuid4())[:5]
    #timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    #exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
    #log_dir = osp.join(default_log_dir, exp_name)

    # Set up logging
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    exp_count = 1
    exp_name = "%s_%s_%04d" % ('experiment', timestamp, exp_count)
    log_dir = config.LOG_DIR + "/local/" + "test/"  + exp_name

    tabular_log_file = osp.join(log_dir, 'progress.csv')
    text_log_file = osp.join(log_dir, 'debug.log')

    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    logger.push_prefix("[%s] " % exp_name)


def get_policy(file):
    policy = None
    train_env = None
    if ':' in args.file:
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


if __name__ == "__main__":

    # With no action limits, this policy can move from any initial position (low initial velocity) to any goal within a large goal region.
    # To test something on the robot, this would be the best.
    pkl_file = "upload/fine_tune/train139/params.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=pkl_file,
                        help='path to the snapshot file')
    parser.add_argument('--max_length', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Speedup')
    parser.add_argument('--num_goals', type=int, default=100,
                        help='Number of test goals')
    args = parser.parse_args()

    policy, train_env = get_policy(args.file)

    # Add one to account for the goal created during environment initialization.
    # TODO - fix this hack.
    test_goal_generator = PR2CrownGoalGeneratorSmall()
        # small_range=False,
        # num_test_goals=args.num_goals + 1,
        # seed=0)

    action_limiter = FixedActionLimiter(0.1)

    env = normalize(Pr2EnvReal(
        goal_generator=test_goal_generator,
        action_limiter=action_limiter,
        distance_thresh=0.01,  # 1 cm
        model="pr2_legofree.xml",
    ))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have n hidden layers, each with k hidden units.
        hidden_sizes=(64, 64, 64),
        # output_gain=1,
        init_std=1,
        trainable=False,
#       pkl_path="/home/ignasi/GitRepos/rllab-private/data/s3/train-Lego/state/random_random_pixel_penalty_p0005_d_06_reward_distance_1_angle_02/params.pkl",
        json_path="~/data/data_ported/random_random_pixel_penalty_p0005_d_06_reward_distance_1_angle_02_right_goal_normal_sample/params.json",
    )

    #setup_logging()
    do_test(env, policy, args.num_goals, args.max_length)
