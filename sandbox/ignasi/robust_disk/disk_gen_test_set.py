import os
import random
import os.path as osp

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import sys
from multiprocessing import cpu_count
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator
from sandbox.carlos_snn.autoclone import autoclone
from rllab import config

import random
import numpy as np

from rllab.misc import logger
from collections import OrderedDict
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.young_clgan.state.evaluator import convert_label, label_states, evaluate_states, label_states_from_paths
from sandbox.young_clgan.envs.base import UniformListStateGenerator, FixedStateGenerator
from sandbox.young_clgan.state.utils import StateCollection, SmartStateCollection

from sandbox.young_clgan.envs.start_env import generate_starts, find_all_feasible_states
from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.ignasi.robust_disk.envs.disk_generate_states_env import DiskGenerateStatesEnv

"""
Generates the test set
"""

def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    inner_env = normalize(DiskGenerateStatesEnv(kill_peg_radius=v['kill_peg_radius'], kill_radius=v['kill_radius']))

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    fixed_start_generator = FixedStateGenerator(state=v['ultimate_goal'])

    gen_states_env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=fixed_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[-1 * v['goal_size']:], # changed!
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        inner_weight=v['inner_weight'],
        goal_weight=v['goal_weight'],
        terminate_env=True,
        append_goal_to_observation = False, # prevents goal environment from appending observation
    )


    with gen_states_env.set_kill_outside():
        seed_starts = generate_starts(gen_states_env, starts=[v['start_goal']], horizon=v['brownian_horizon'], animated=False,
                                      variance=v['brownian_variance'], subsample=v['num_new_starts'])  # , animated=True, speedup=1)

        find_all_feasible_states(gen_states_env, seed_starts, distance_threshold=0.1,
                                 brownian_variance=1, animate=False, max_states=v['max_gen_states'],
                                 horizon=500,
                                 # states_transform= states_transform
                                 )
        import sys; sys.exit(0)

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
    subnets = [
        # "us-east-2a", "us-east-2b",
        'us-east-1a', 'us-east-1d', 'us-east-1e'
    ]
    # subnets = [
    #     'ap-northeast-2a', 'ap-northeast-2c', 'us-east-2b', 'ap-south-1a', 'us-east-2c', 'us-east-2a', 'ap-south-1b',
    #     'us-east-1b', 'us-east-1a', 'us-east-1d', 'us-east-1e', 'eu-west-1c', 'eu-west-1a', 'eu-west-1b'
    # ]
    ec2_instance = args.type if args.type else 'c4.4xlarge'
    # configure instan
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = cpu_count() if not args.debug else 1
    else:
        mode = 'local'
        n_parallel = cpu_count() if not args.debug else 1
        # n_parallel = multiprocessing.cpu_count()


    vg = VariantGenerator()
    vg.add('start_size', [9])

    # changed
    vg.add('start_goal', [(1.42616709, -0.01514247, 2.64956251, -1.88308175, 4.79495397, -1.05910442, -1.339634, 0, 0)])  # added two coordinates
    # vg.add('start_goal', [(1.42616709, -0.01514247, 2.64956251, -1.88308175, 4.79495397, -1.05910442, -1.339634, 0.4146814, 0.47640087)]) # added two coordinates
    vg.add('ultimate_goal', [(0.4146814, 0.47640087, 0.5305665)])
    vg.add('goal_size', [3]) # changed
    vg.add('terminal_eps', [0.03])
    # brownian params
    # vg.add('seed_with', ['on_policy', 'only_goods', 'all_previous'])  # good from brown, onPolicy, previousBrown (ie no good)
    vg.add('seed_with', ['only_goods'])  # good from brown, onPolicy, previousBrown (ie no good)
    # vg.add('brownian_horizon', lambda seed_with: [0, 50, 500] if seed_with == 'on_policy' else [50, 500])
    vg.add('brownian_horizon', [300])
    vg.add('brownian_variance', [2])
    vg.add('regularize_starts', [0])
    # goal-algo params
    vg.add('min_reward', [0.1])
    vg.add('max_reward', [0.9])
    vg.add('distance_metric', ['L2'])
    vg.add('extend_dist_rew', [False])
    vg.add('inner_weight', [0])
    vg.add('goal_weight', lambda inner_weight: [1000] if inner_weight > 0 else [1])
    vg.add('persistence', [1])
    vg.add('n_traj', [3])  # only for labeling and plotting (for now, later it will have to be equal to persistence!)
    vg.add('with_replacement', [True])
    vg.add('use_trpo_paths', [True])
    # replay buffer

    vg.add('replay_buffer', [True])  # todo: attention!!
    vg.add('coll_eps', lambda terminal_eps: [terminal_eps])
    vg.add('num_new_starts', [200])
    vg.add('num_old_starts', [100])
    vg.add('smart_replay_buffer', [True])
    # vg.add('smart_replay_buffer', [True])
    vg.add('smart_replay_abs', [True])
    # vg.add('smart_replay_abs', [True, False])
    # vg.add('smart_replay_eps', [0.2, 0.5, 1])
    vg.add('smart_replay_eps', [0.5])
    # vg.add('smart_replay_eps', [1.0])  # should break
    # sampling params
    vg.add('horizon', [500])
    vg.add('outer_iters', [5000])
    vg.add('inner_iters', [5])  # again we will have to divide/adjust the
    vg.add('pg_batch_size', [100000])
    # policy initialization
    vg.add('output_gain', [0.1])
    vg.add('policy_init_std', [1])
    vg.add('learn_std', [False])
    vg.add('adaptive_std', [False])
    vg.add('discount', [0.995])
    vg.add('baseline', ["g_mlp"])
    # vg.add('policy', ['recurrent'])
    vg.add('policy', ['mlp'])
    # vg.add('policy', ['recurrent', 'mlp'])

    # vg.add('seed', range(100, 600, 100))
    vg.add('seed', [13,23,33])

    vg.add('generating_test_set', [False]) #TODO can change
    vg.add('move_peg', [True]) # whether or not to move peg
    vg.add('kill_radius', [0.5])
    vg.add('kill_peg_radius', [0.05])
    vg.add('max_gen_states', [50000])
    # vg.add('peg_positions', [(7,8)])  # joint numbers for peg
    # vg.add('peg_scaling', [10]) # multiplicative factor to peg position

    # exp_prefix = "robust-disk-test2"
    # exp_prefix = 'uniform200-mass300000'
    exp_prefix = "test50000"
    # Launching
    print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(exp_prefix, vg.size))
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)
    mode = "ec2"
    mode="local"
    for vv in vg.variants():

        # run_task(vv)
        run_experiment_lite(
            # use_cloudpickle=False,
            stub_method_call=run_task,
            variant=vv,
            mode='local',
            n_parallel=3,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=vv['seed'],
            exp_prefix=exp_prefix,
            plot=True,
            # exp_name=exp_name,
        )
