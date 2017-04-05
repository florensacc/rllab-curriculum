import sys
import os
import os.path as osp
import argparse
import random
import numpy as np

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Symbols that need to be stubbed
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import rllab.misc.logger
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.stateful_pool import singleton_pool
from rllab.envs.box2d.pendulum_env import PendulumEnv
from rllab import config
from rllab.misc.instrument import VariantGenerator

from sandbox.carlos_snn.init_sampler.base import UniformInitGenerator, UniformListInitGenerator, FixedInitGenerator
from sandbox.carlos_snn.init_sampler.base import InitExplorationEnv
from sandbox.young_clgan.lib.logging import *
from sandbox.carlos_snn.autoclone import autoclone

from sandbox.young_clgan.experiments.point_env_maze.maze_evaluate import test_and_plot_policy
from sandbox.young_clgan.lib.envs.maze.point_maze_env import PointMazeEnv
from sandbox.young_clgan.lib.goal.utils import GoalCollection
from sandbox.young_clgan.lib.logging import HTMLReport
from sandbox.young_clgan.lib.logging import format_dict
from sandbox.young_clgan.lib.logging.visualization import save_image, plot_gan_samples, plot_labeled_samples, \
    plot_line_graph
from sandbox.young_clgan.lib.envs.base import UniformListGoalGenerator, FixedGoalGenerator, update_env_goal_generator, \
    generate_initial_goals, UniformGoalGenerator
from sandbox.young_clgan.lib.goal import *
# from sandbox.young_clgan.lib.logging import *
# from sandbox.young_clgan.lib.logging.logger import ExperimentLogger

from sandbox.young_clgan.lib.logging.logger import ExperimentLogger, AttrDict, format_experiment_log_path, make_log_dirs
from sandbox.young_clgan.lib.goal.evaluator import convert_label, evaluate_goal_env
from rllab.misc import logger

# from sandbox.young_clgan.lib.utils import initialize_parallel_sampler
# initialize_parallel_sampler()


EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]

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
    args = parser.parse_args()

    if args.clone:
        autoclone.autoclone(__file__, args)

    # setup ec2
    subnets = [
        'us-east-2b', 'us-east-1a', 'us-east-1d', 'us-east-1b', 'us-east-1e', 'ap-south-1b', 'ap-south-1a', 'us-west-1a'
    ]
    ec2_instance = args.type if args.type else 'c4.4xlarge'

    # configure instance
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = 4
    else:
        mode = 'local'
        n_parallel = 1
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    exp_prefix = 'init-maze-trpo'
    vg = VariantGenerator()
    # algorithm params
    vg.add('seed', range(30, 60, 10))
    vg.add('n_itr', [500])
    vg.add('batch_size', [5000])
    vg.add('max_path_length', [200])
    # environemnt params
    vg.add('init_generator', [UniformInitGenerator])
    vg.add('init_center', [(2,2)])
    vg.add('init_range', lambda init_generator: [3] if init_generator == UniformInitGenerator else [None])
    vg.add('angle_idxs', lambda init_generator: [(None,)])
    vg.add('goal', [(0, 4), ])
    vg.add('final_goal', lambda goal: [goal])
    vg.add('goal_reward', ['NegativeDistance'])
    vg.add('goal_weight', [0])  # this makes the task spars
    vg.add('terminal_bonus', [1])
    vg.add('reward_dist_threshold', [0.3])
    vg.add('terminal_eps', lambda reward_dist_threshold: [reward_dist_threshold])
    vg.add('indicator_reward', [True])
    vg.add('outer_iter', [500])
    # policy hypers
    vg.add('learn_std', [True])
    vg.add('policy_init_std', [1])
    vg.add('output_gain', [1])


    def run_task(v):
        random.seed(v['seed'])
        np.random.seed(v['seed'])

        # tf_session = tf.Session()

        # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
        logger.log("Initializing report and plot_policy_reward...")
        log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
        report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=2)
        report.add_header("{}".format(EXPERIMENT_TYPE))
        report.add_text(format_dict(v))

        inner_env = normalize(PointMazeEnv(
            goal_generator=FixedGoalGenerator(v['final_goal']),
            reward_dist_threshold=v['reward_dist_threshold'],
            indicator_reward=v['indicator_reward'],
            terminal_eps=v['terminal_eps'],
        ))

        init_generator_class = v['init_generator']
        if init_generator_class == UniformInitGenerator:
            init_generator = init_generator_class(init_size=np.size(v['goal']), bound=v['init_range'], center=v['init_center'])
        else:
            assert init_generator_class == FixedInitGenerator, 'Init generator not recognized!'
            init_generator = init_generator_class(goal=v['goal'])

        env = InitExplorationEnv(env=inner_env, goal=v['goal'], init_generator=init_generator, goal_reward=v['goal_reward'],
                                 goal_weight=v['goal_weight'], terminal_bonus=v['terminal_bonus'], angle_idxs=v['angle_idxs'])

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            # Fix the variance since different goals will require different variances, making this parameter hard to learn.
            learn_std=v['learn_std'],
            output_gain=v['output_gain'],
            init_std=v['policy_init_std'],
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        n_traj = 3
        sampling_res = 2
        # report.save()
        report.new_row()

        all_mean_rewards = []
        all_coverage = []
        all_success = []

        # for outer_iter in range(v['outer_iters']):
        #
        #     goals = np.random.uniform(-v['goal_range'], v['goal_range'], size=(300, v['goal_size']))

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=v['batch_size'],
            max_path_length=v['max_path_length'],
            n_itr=v['n_itr'],
            discount=0.99,
            step_size=0.01,
            plot=False,
        )

        algo.train()


    for vv in vg.variants(randomized=True):

        if mode in ['ec2', 'local_docker']:
            # # choose subnet
            # subnet = random.choice(subnets)
            # config.AWS_REGION_NAME = subnet[:-1]
            # config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
            #     config.AWS_REGION_NAME]
            # config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
            #     config.AWS_REGION_NAME]
            # config.AWS_SECURITY_GROUP_IDS = \
            #     config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
            #         config.AWS_REGION_NAME]
            # config.AWS_NETWORK_INTERFACES = [
            #     dict(
            #         SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
            #         Groups=config.AWS_SECURITY_GROUP_IDS,
            #         DeviceIndex=0,
            #         AssociatePublicIpAddress=True,
            #     )
            # ]

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
                # # use this ONLY with ec2 or local_docker!!!
                pre_commands=[
                    "pip install --upgrade pip",
                    "pip install --upgrade theano"
                ],
            )
            # if mode == 'local_docker':
            #     sys.exit()
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
                # exp_name=exp_name,
            )

