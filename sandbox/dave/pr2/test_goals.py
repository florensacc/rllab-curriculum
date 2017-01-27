import argparse

import datetime
import os
import uuid

import dateutil
import joblib

import rllab.misc.logger as logger
from sandbox.dave.pr2.action_limiter import CurriculumActionLimiter, FixedActionLimiter
from rllab import config

from rllab.envs.normalized_env import normalize
import os.path as osp

from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2CrownGoalGeneratorSmall #PR2BoxGoalGeneratorSmall #PR2FixedGoalGenerator #PR2CrownGoalGeneratorSmall #PR2TestGoalGenerator
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoBoxBlockGeneratorLarge #PR2LegoBoxBlockGeneratorSmall #PR2LegoFixedBlockGenerator #PR2TestGoalGenerator
# from sandbox.dave.rllab.policies.pretrain_gaussian_mlp_policy import PretrainGaussianMLPPolicy

from sandbox.dave.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.dave.rllab.policies.gaussian_mlp_policy_tanh import GaussianMLPPolicy

# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_position import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_hand import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_reach import Pr2EnvLego
from rllab.sampler.utils import rollout
from sandbox.dave.utils.ploting import *

filename = str(uuid.uuid4())


def do_test(env, policy, num_test_goals, max_path_length):
    paths = []
    for itr in range(num_test_goals):
        with logger.prefix('itr #%d | ' % itr):
            path = rollout(env, policy, animated=True, max_path_length=max_path_length, speedup=10)
            paths.append(path)
            env.log_diagnostics([path])
            policy.log_diagnostics([path])
            print('iteration:     ', itr)
    plot_heatmap(paths, 'lego')


            #loggerdump_tabular(with_prefix=True)
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
    log_dir = config.LOG_DIR + "/local/" + "test/" + exp_name

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

    # With no action limits, this policy can move from any initial position and velocity to any goal within a small goal region.
    # To test something on the robot, this would be the best.
    #pkl_file = "data/s3/train96/train96_2016_08_25_15_18_44_0021/params.pkl"

    # With no action limits, this policy can move from any initial position (low initial velocity) to any goal within a large goal region.
    # To test something on the robot, this would be the best.
    #pkl_file = "data/s3/train139/train139_2016_09_01_15_59_53_0001/params.pkl"

    # pkl_file = "/home/ignasi/GitRepos/rllab-private/data/s3/train-Lego/RSS/position-control-01/position-control-01-lego-gravity-eve-fix1/params.pkl"
    # pkl_file = "/home/ignasi/GitRepos/rllab-private/data/s3/train-Lego/RSS/position-control-random-lego/position-control-random-lego-1/params.pkl"
    # pkl_file = "/home/ignasi/GitRepos/rllab-private/data/s3/train-Lego/RSS/reward-guiding-0999/reward-guiding-09991/params.pkl"
    # pkl_file = "/home/ignasi/GitRepos/rllab-private/data/s3/train-Lego/RSS/position-control-random-everything/position_control_random_everything1/params.pkl"
    # pkl_file = "/home/ignasi/GitRepos/rllab-private/data/local/train-Lego/RSS/debug-beta-02/debug-beta-0221/params.pkl"
    # pkl_file = "/home/ignasi/GitRepos/rllab-private/data/s3/train-Lego/RSS/debug-constant-time/debug-constant-time21/params.pkl"
    # pkl_file = "/home/ignasi/GitRepos/rllab-private/data/s3/train-Lego/RSS/stds-random-lego-mean-gain-001/std-0005-good-random-lego-mean-gain-0010.1/params.pkl"
    # pkl_file = "data/local/train-Lego/RSS/position-control-fine-tune/position_control_fine_tune1/params.pkl"
    # pkl_file = "data/s3/train-Lego/RSS/comparision-torque-position/position-control-n05-arm-fixed-lego//params.pkl"
    # pkl_file = "data/s3/train-Lego/RSS/fixed-arm-position-ctrl-tip/fixed-arm-position-ctrl-tip1/params.pkl"
    # pkl_file = "data/s3/train-Lego/RSS/fixed-arm-position-ctrl-tip/fixed-arm-position-ctrl-tip1/params.pkl"
    pkl_file = "data/s3/train-Lego/RSS/fixed-arm-position-ctrl-tip-no-random-restarts/fixed-arm-position-ctrl-tip-no-random-restarts1/params.pkl"
    # pkl_file = "data/local/train-Lego/RSS/baseline/training/params.pkl"





    parser = argparse.ArgumentParser()
    # parser.add_argument('--file', type=str, default=pkl_file,
    #                     help='path to the snapshot file')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Speedup')
    parser.add_argument('--num_goals', type=int, default=np.int(np.square(0.4/0.01)),
                        help='Number of test goals')
    args = parser.parse_args()

    # policy, train_env = get_policy(args.file)

    # Add one to account for the goal created during environment initialization.
    # TODO - fix this hack.
    test_goal_generator = PR2CrownGoalGeneratorSmall() #PR2TestGoalGenerator()  #PR2TestGoalGenerator(
    test_lego_generator = PR2LegoBoxBlockGeneratorLarge()

    # env = normalize(Pr2Env(
    #     goal_generator=test_goal_generator,
    #     allow_random_restarts=True,
    #     allow_random_vel_restarts=True,
    #     qvel_init_std=0.01,
    #     pos_normal_sample=True,
    #     pos_normal_sample_std=0.01,
    #     max_action=0.1,
    #     model="pr2_1arm_g.xml",
    # ))

    # env = normalize(Pr2Env(
    #     goal_generator=test_goal_generator,
    #     allow_random_restarts=True,
    #     allow_random_vel_restarts=True,
    #     #qvel_init_std=0.01,
    #     # pos_normal_sample=True,
    #     # pos_normal_sample_std=0.01,
    #     #max_action=0.1,
    #     model="pr2_1arm.xml",
    #     #model="pr2_1arm.xml",
    # ))

    # action_limiter = CurriculumActionLimiter(
    #     update_delta=0.1,
    #     target_paths_within_thresh=0.96
    # )

    action_limiter = FixedActionLimiter(3)

    env = normalize(Pr2EnvLego(
        goal_generator=test_goal_generator,
        lego_generator=test_lego_generator,
        action_limiter=action_limiter,
        allow_random_restarts=True,
        allow_random_vel_restarts=False,
        distance_thresh=0.01,  # 1 cm
        qvel_init_std=0.01,
        pos_normal_sample=True, # Uniform sampling
        pos_normal_sample_std=0.5,
        # model="pr2_legofree.xml",
        use_vision=True,
        crop=True,
        # beta=0.1,
        # number_actions=5
        # use_depth=True,
    ))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have n hidden layers, each with k hidden units.
        # hidden_sizes=(64, 64, 64),
        # output_gain=1,
        init_std=0,
        # beta=0.1,
        # pkl_path="/home/ignasi/GitRepos/rllab-private/data/s3/train-Lego/state/random_random_pixel_penalty_p0005_d_06_reward_distance_1_angle_02_crown_normal_sample_001_50000/params.pkl"
        # json_path=json_path,
        # npz_path=npz_path,
        pkl_path=pkl_file,
        )


    # policy = ScaledGaussianMLPPolicy(
    #     env_spec=env.spec,
    #     # The neural network policy should have n hidden layers, each with k hidden units.
    #     hidden_sizes=(64, 64, 64),
    #     warm_pkl_path=args.file,
    # )

    #setup_logging()
    do_test(env, policy, args.num_goals, args.max_length)

