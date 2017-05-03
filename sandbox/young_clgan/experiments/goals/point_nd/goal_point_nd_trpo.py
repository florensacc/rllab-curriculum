from sandbox.young_clgan.utils import set_env_no_gpu, format_experiment_prefix
set_env_no_gpu()

import argparse
import math
import os
import os.path as osp
import sys
import random
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator

from sandbox.carlos_snn.autoclone import autoclone

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from sandbox.young_clgan.envs.ndim_point.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.young_clgan.envs.goal_env import GoalExplorationEnv, update_env_goal_generator, \
    evaluate_goal_env
from sandbox.young_clgan.envs.base import FixedStateGenerator, UniformStateGenerator, \
    update_env_state_generator

from sandbox.young_clgan.state.evaluator import *
from sandbox.young_clgan.logging.html_report import format_dict, HTMLReport
from sandbox.young_clgan.logging.visualization import *
from sandbox.young_clgan.logging.logger import ExperimentLogger

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    tf_session = tf.Session()

    # goal generators
    logger.log("Initializing the goal generators and the inner env...")
    inner_env = normalize(PointEnv(dim=v['goal_size'], state_bounds=v['state_bounds']))
    print("the state_bounds are: ", v['state_bounds'])

    center = np.zeros(v['goal_size'])
    fixed_goal_generator = FixedStateGenerator(center)
    uniform_goal_generator = UniformStateGenerator(state_size=v['goal_size'], bounds=v['goal_range'],
                                                  center=center)
    feasible_goal_ub = np.array(v['state_bounds'])[:v['goal_size']]
    print("the feasible_goal_ub is: ", feasible_goal_ub)
    uniform_feasible_goal_generator = UniformStateGenerator(state_size=v['goal_size'], bounds=[-1 * feasible_goal_ub,
                                                                                             feasible_goal_ub])

    env = GoalExplorationEnv(
        env=inner_env, goal_generator=uniform_goal_generator,
        obs_transform=lambda x:x[:int(len(x) / 2)],
        dist_threshold=v['reward_dist_threshold'],
        distance_metric=v['distance_metric'],
        terminate_env=True, goal_weight=v['goal_weight'],
    )  # this goal_generator will be updated by a uniform after
    
    if v['sample_unif_feas']:
        env.update_goal_generator(uniform_feasible_goal_generator)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        learn_std=False,
        output_gain=v['output_gain'],
        init_std=v['policy_init_std'],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    n_traj = 3

    # feasible_goals = generate_initial_goals(env, policy, v['goal_range'], horizon=v['horizon'], size=10000) #v['horizon'])
    # print(feasible_goals)
    # uniform_list_goal_generator = UniformListStateGenerator(goal_list=feasible_goals.tolist())
    # env.update_goal_generator(uniform_list_goal_generator)

    # env.update_goal_generator(fixed_goal_generator)

    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)
    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    logger.log("Starting the outer iterations")
    for outer_iter in range(v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        
        

        logger.log("Perform TRPO with UniformListStateGenerator...")
        with ExperimentLogger(log_dir, outer_iter, snapshot_mode='last', hold_outter_log=True):
            # set goal generator to uniformly sample from selected all_goals
            # update_env_state_generator(
            #     env,
            #     UniformListStateGenerator(
            #         goals.tolist()
            #     )
            # )

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=v['pg_batch_size'],
                max_path_length=v['horizon'],
                n_itr=v['inner_iters'],
                discount=0.995,
                step_size=0.01,
                plot=False,
            )

            algo.train()

        

        # log some more on how the pendulum performs the upright and general task
        old_goal_generator = env.goal_generator
        logger.log("Evaluating performance on Unif and Fix Goal Gen...")
        with logger.tabular_prefix('UnifFeasGoalGen_'):
            update_env_state_generator(env, uniform_feasible_goal_generator)
            evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=50,
                              fig_prefix='UnifFeasGoalGen_itr%d' % outer_iter,
                              report=report, n_traj=n_traj)
        # back to old goal generator
        with logger.tabular_prefix("UnifGoalGen_"):
            update_env_state_generator(env, old_goal_generator)
            evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=50,
                              fig_prefix='UnifGoalGen_itr%d' % outer_iter,
                              report=report, n_traj=n_traj)
        # with logger.tabular_prefix('FixGoalGen_'):
        #     update_env_state_generator(env, goal_generator=fixed_goal_generator)
        #     evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5, fig_prefix='FixGoalGen',
        #                       report=report)
        logger.dump_tabular(with_prefix=False)

        report.save()
        report.new_row()

    with logger.tabular_prefix('FINALUnifFeasGoalGen_'):
        update_env_state_generator(env, uniform_feasible_goal_generator)
        evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5e3, fig_prefix='FINAL1UnifFeasGoalGen_',
                          report=report, n_traj=n_traj)
        evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5e3, fig_prefix='FINAL2UnifFeasGoalGen_',
                          report=report, n_traj=n_traj)
    logger.dump_tabular(with_prefix=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ec2', '-e', action='store_true', default=False, help="add flag to run in ec2")
    parser.add_argument('--clone', '-c', action='store_true', default=False,
                        help="add flag to copy file and checkout current")
    parser.add_argument('--local_docker', '-d', action='store_true', default=False,
                        help="add flag to run in local dock")
    parser.add_argument('--type', '-t', type=str, default='', help='set instance type')
    parser.add_argument('--price', '-p', type=str, default='', help='set betting price')
    parser.add_argument('--subnet', '-sn', type=str, default='', help='set subnet like us-west-1a')
    parser.add_argument('--name', '-n', type=str, default='', help='set exp prefix name and new file name')
    parser.add_argument('--debug', action='store_true', default=False, help="run code without multiprocessing")
    args = parser.parse_args()

    if args.clone:
        autoclone.autoclone(__file__, args)

    # setup ec2
    ec2_instance = args.type if args.type else 'm4.4xlarge'

    # configure instance
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"])  # make the default 4 if not using ec2
    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = cpu_count() if not args.debug else 1
    else:
        mode = 'local'
        n_parallel = cpu_count() if not args.debug else 1

    exp_prefix = format_experiment_prefix('goal-point-nd-trpo')
    vg = VariantGenerator()

    vg.add('seed', range(30, 90, 20))
    # # GeneratorEnv params
    vg.add('goal_size', [6, 5, 4, 3, 2])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('goal_range', [5])  # this will be used also as bound of the state_space
    vg.add('sample_unif_feas', [True])
    vg.add('reward_dist_threshold', lambda goal_size: [math.sqrt(goal_size) / math.sqrt(2) * 0.3])
    # vg.add('angle_idxs', [((0, 1),)]) # these are the idx of the obs corresponding to angles (here the first 2)
    vg.add('distance_metric', ['L2'])
    vg.add('goal_weight', [300])
    vg.add('state_bounds', lambda goal_range, goal_size, reward_dist_threshold:
    [(1, goal_range) + (0.3,) * (goal_size - 2) + (goal_range, ) * goal_size])
    #############################################
    vg.add('min_reward', lambda goal_weight: [goal_weight * 0.1])  # now running it with only the terminal reward of 1!
    vg.add('max_reward', lambda goal_weight: [goal_weight * 0.9])
    vg.add('horizon', [200])
    vg.add('outer_iters', [400])
    vg.add('inner_iters', [5])
    vg.add('pg_batch_size', [20000])
    # policy initialization
    vg.add('output_gain', [1])
    vg.add('policy_init_std', [1])

    print('Running {} inst. on type {}, with price {}, parallel {}'.format(
        vg.size, config.AWS_INSTANCE_TYPE,
        config.AWS_SPOT_PRICE, n_parallel
    ))

    for vv in vg.variants():

        if mode in ['ec2', 'local_docker']:

            run_experiment_lite(
                # use_cloudpickle=False,
                stub_method_call=run_task,
                variant=vv,
                mode=mode,
                # Number of parallel workers for sampling
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=vv['seed'],
                # plot=True,
                exp_prefix=exp_prefix,
                # exp_name=exp_name,
                sync_s3_pkl=True,
                # for sync the pkl file also during the training
                sync_s3_png=True,
                sync_s3_html=True,
                # # use this ONLY with ec2 or local_docker!!!
                pre_commands=[
                    'export MPLBACKEND=Agg',
                    'pip install --upgrade pip',
                    'pip install --upgrade -I tensorflow',
                    'pip install git+https://github.com/tflearn/tflearn.git',
                    'pip install dominate',
                    'pip install scikit-image',
                    'conda install numpy -n rllab3 -y',
                ],
            )
            if mode == 'local_docker':
                sys.exit()
        else:
            run_experiment_lite(
                # use_cloudpickle=False,
                stub_method_call=run_task,
                variant=vv,
                mode='local',
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=vv['seed'],
                exp_prefix=exp_prefix,
                print_command=False,
            )
            if args.debug:
                sys.exit()
