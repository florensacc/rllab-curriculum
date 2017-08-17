import sys
from rllab.misc import logger
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.logging.visualization import plot_labeled_states


from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from sandbox.dave.rllab.algos.trpo import TRPO
# from rllab.algos.trpo import TRPO
from sandbox.dave.rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.envs.normalized_env import normalize
from sandbox.dave.rllab.goal_generators.goal_generator import FixedGoalGenerator
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2CrownGoalGeneratorSmall, PR2FixedGoalGenerator #PR2CrownGoalGeneratorSmall
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoBoxBlockGeneratorSmall, PR2LegoBoxBlockGeneratorSmall,PR2LegoBoxBlockGeneratorSmall, PR2LegoFixedBlockGenerator
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.dave.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.dave.rllab.policies.gaussian_mlp_policy_tanh import GaussianMLPPolicy
from rllab.misc.instrument import VariantGenerator, variant

from sandbox.young_clgan.envs.block_pushing.pr2_env_lego_hand import Pr2EnvLego
from rllab import config
import os
import random
import os.path as osp
import numpy as np
from sandbox.young_clgan.envs.start_env import generate_starts
from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.young_clgan.envs.block_pushing.pushing_evaluate import test_and_plot_policy, plot_policy_means, plot_heatmap, plot_pushing
from sandbox.young_clgan.logging.visualization import save_image


from sandbox.young_clgan.state.evaluator import label_states
from sandbox.young_clgan.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from sandbox.young_clgan.state.generator import StateGAN
from sandbox.young_clgan.state.utils import StateCollection
# stub(globals())
#TODO: figure out crown goal generator

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]

def run_task(v):
    # for inner environment, goal_generator shouldn't do anything and lego_generator shouldn't do anything
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    #These are used exactly twice--when mujoco_env is initialized and calls reset
    # i think the correct way to avoid using them is to modify the reset called by init of mujoco env
    goal_generator = PR2FixedGoalGenerator(goal = (0.6, 0.1, 0.5025)) # second dimension moves block further away vertically
    lego_generator = PR2LegoFixedBlockGenerator(block = (0.6, 0.2, 0.5025, 1, 0, 0, 0)) # want block at 0.6 +/- 0.2, , 0.1 +/- 0.4, 0.5025

    # plotting code, similar to maze/starts
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    if log_dir is None: #need to use run_experiment_lite for log_dir not to be none
        log_dir = "/home/michael/"
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=1)
    logger.log(osp.join(log_dir, 'report.html'))
    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    # relevant positions for environment
    init_hand = np.array(v['init_hand'])
    fixed_goal_generator = FixedStateGenerator(state=v['lego_target'])
    uniform_start_generator = UniformStateGenerator(state_size=3, bounds=(v['lego_init_lower'], v['lego_init_upper']),
                                                    center=v['lego_init_center'])
    action_limiter = FixedActionLimiter()

    inner_env = normalize(Pr2EnvLego(
        goal_generator=goal_generator,
        lego_generator=lego_generator,
        max_action=1,
        pos_normal_sample=True,
        action_limiter=action_limiter,
        qvel_init_std=0, #0.01,
        # pos_normal_sample_std=.01, # ignored i think?
        fixed_target = init_hand, # sets the initial position of the hand to 0.6 0.3
        reward_function=v['reward_function'],
        gamma=v['discount'],
        phi_positive=v['phi_positive']
        # allow_random_restarts=True, #ignored i think?
    ))

    env = GoalStartExplorationEnv(
        env=inner_env,
        append_start=False,
        start_generator=uniform_start_generator,
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[-3:],
        obs2start_transform=lambda x: x[-3:],
        # start is just the initial lego position
        terminal_eps = v['terminal_eps'],  # TODO: potentially make more lenient?
        distance_metric = 'L2',
        extend_distance_rew = v['extend_distance_rew'],  # I think this turns off L2 distance reward
        # distance_rew = True, # check, I think this checks the desired distance
        terminate_env = v['terminate_env'],
        inner_weight = v['inner_weight'],
        goal_weight = v['goal_weight'],
    )

    # Follows Ignasi's code
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have n hidden layers, each with k hidden units.
        hidden_sizes=(64, 64, 64),
        init_std=v['init_std'],
        output_gain=0.1,
        )

    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']
    outer_iter = 0
    logger.log("Generating the initial heat map...")
    plot_pushing(policy, env, report, bounds=(v['lego_init_lower'], v['lego_init_upper']),
                                                    center=v['lego_init_center'], itr=outer_iter)


    baseline = LinearFeatureBaseline(env_spec=env.spec)
    for outer_iter in range(1, v['outer_iters']):
        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Outer itr # %i" % outer_iter)
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=v['batch_size'],
                max_path_length=v['max_path_length'],  #100 #making path length longer is fine because of early termination
                n_itr=v['inner_iters'],
                discount=v['discount'],
                gae_lambda=0.98, #todo: can consider changing
                step_size=0.01,
                # goal_generator=goal_generator,
                action_limiter=None,
                optimizer_args={'subsample_factor': 0.1},
                )

            algo.train()

            plot_pushing(policy, env, report, bounds=(v['lego_init_lower'], v['lego_init_upper']),
                             center=v['lego_init_center'], itr=outer_iter)
            report.save()


vg = VariantGenerator()
# vg.add('seed', [1])
vg.add('seed', [2,12,22])

# Environment parameters
vg.add('init_hand', [[0.6,  0.35,  0.5025],]) # changed
vg.add('lego_target', [(0.6, 0.12, 0.5025),])
vg.add('lego_init_center', [(0.6, 0.15, 0.5025),])
vg.add('lego_init_lower', [(-0.1, -0.02, 0),])
vg.add('lego_init_upper', [(0.1, 0.1, 0),])
vg.add('reward_function', ["dense"])
# vg.add('reward_function', ["shaped", "dense"]) #TODO: change
# vg.add('phi_positive', [False, True]) #TODO: change
vg.add('phi_positive', [False]) #TODO: changed

# Network parameters
vg.add('init_std', [0.1])

# Optimizer parameters
vg.add('inner_iters', [5])
vg.add('outer_iters', [800])
vg.add('batch_size', [15000])
vg.add('max_path_length', [80])
vg.add('discount', [0.95])

# Goal generation parameters
vg.add('terminal_eps', [0.05])
vg.add('extend_distance_rew', [False])
vg.add('terminate_env', [True]) #TODO: maybe do not allow to terminate early??

# all the reward is from the inner environment
vg.add('inner_weight', [1])
vg.add('goal_weight', [0])

mode = "local"
# mode = "ec2"
ec2_instance = 'c4.4xlarge'
info = config.INSTANCE_TYPE_INFO[ec2_instance]
config.AWS_INSTANCE_TYPE = ec2_instance
config.AWS_SPOT_PRICE = str(info["price"])
n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2

for vv in vg.variants():
    # run_task(vv) # uncomment when debugging
    if mode in ['ec2', "docker"]:

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
            exp_prefix="handbenchmark5",
            # exp_name=exp_name,
            # for sync the pkl file also during the training
            sync_s3_pkl=True,
            # sync_s3_png=True,
            sync_s3_html=True,
            use_cloudpickle=False,
            # # use this ONLY with ec2 or local_docker!!!
            pre_commands=[
                'export MPLBACKEND=Agg',
                'pip install --upgrade pip',
                'pip install --upgrade -I tensorflow',
                'pip install git+https://github.com/tflearn/tflearn.git',
                'pip install dominate',
                'pip install multiprocessing_on_dill',
                'pip install scikit-image',
                'conda install numpy -n rllab3 -y',
            ],
        )

        sys.exit()
    else:
        run_experiment_lite(
            stub_method_call=run_task,
            use_gpu=False,
            variant=vv,
            # Number of parallel workers for sampling
            # n_parallel=32,
            n_parallel=2, # use cpu_count in the future
            snapshot_mode="last",
            # snapshot_mode="gap",
            # snapshot_gap=100,
            seed=vv['seed'],
            mode="local",
            # mode="ec2",
            exp_prefix="handbenchmark16",
            # exp_prefix="handbenchmarkdebug",
            # exp_name= "decaying-decaying-gamma" + str(t),
            # plot=True,
        )
        sys.exit()


# add more stuff
    # test_and_plot_policy(policy,
    #                      env,
    #                      as_goals=False,
    #                      # max_reward=v['max_reward'], #unused?
    #                      sampling_res=sampling_res,
    #                      n_traj=3,
    #                      itr=outer_iter,
    #                      report=report,
    #                      limit=0.5, # make parameter
    #                      center=v['lego_init_center'][:2],
    #                      bounds=((v['lego_init_lower'][:2], v['lego_init_upper'][:2])),
    #                      )
    # plot_policy_means(policy, env, sampling_res=2, report=report, limit=v['start_range'], center=v['start_center'])s