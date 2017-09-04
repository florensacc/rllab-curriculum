
from rllab.misc import logger
from sandbox.ignasi.logging import HTMLReport
from sandbox.ignasi.logging import format_dict
from sandbox.ignasi.logging.logger import ExperimentLogger
from sandbox.ignasi.logging.visualization import plot_labeled_states


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

from sandbox.ignasi.envs.block_pushing.pr2_env_lego_hand import Pr2EnvLego
from rllab import config
import os
import random
import os.path as osp
import numpy as np
from sandbox.ignasi.envs.start_env import generate_starts, generate_starts_random
from sandbox.ignasi.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.ignasi.envs.block_pushing.pushing_evaluate import test_and_plot_policy, plot_policy_means, plot_heatmap, plot_pushing
from sandbox.ignasi.logging.visualization import save_image


from sandbox.ignasi.state.evaluator import label_states
from sandbox.ignasi.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from sandbox.ignasi.state.generator import StateGAN
from sandbox.ignasi.state.utils import StateCollection



EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]

def run_task(v):
    #TODO: add seed
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    # for inner environment, goal_generator shouldn't do anything and lego_generator shouldn't do anything

    #These are used exactly twice--when mujoco_env is initialized and calls reset
    # i think the correct way to avoid using them is to modify the reset called by init of mujoco env
    goal_generator = PR2FixedGoalGenerator(goal = (0.6, 0.1, 0.5025)) # second dimension moves block further away vertically
    lego_generator = PR2LegoFixedBlockGenerator(block = (0.6, 0.2, 0.5025, 1, 0, 0, 0)) # want block at 0.6 +/- 0.2, , 0.1 +/- 0.4, 0.5025

    # plotting code, similar to maze/starts
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    # need to use run_experiment_lite for log_dir not to be none
    if log_dir is None:
        log_dir = "/home/michael/"
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)
    logger.log(osp.join(log_dir, 'report.html'))
    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    # relevant positions for environment
    init_hand = np.array(v['init_hand'])
    fixed_goal_generator = FixedStateGenerator(state=v['lego_target'])
    uniform_start_generator = UniformStateGenerator(state_size=3, bounds=(v['lego_init_lower'], v['lego_init_upper']),
                                                    center=v['lego_init_center'])

    inner_env = normalize(Pr2EnvLego(
        goal_generator=goal_generator,
        lego_generator=lego_generator,
        max_action=1,
        pos_normal_sample=True,
        qvel_init_std=0, #0.01,
        # pos_normal_sample_std=.01, # ignored i think?
        fixed_target = init_hand, # sets the initial position of the hand to 0.6 0.3
        random_angle=v['random_angle'],
        # allow_random_restarts=True, #ignored i think?
    ))

    env = GoalStartExplorationEnv(
        env=inner_env,
        append_start=False,
        start_generator=uniform_start_generator,
        goal_generator=fixed_goal_generator,
        # obs2goal_transform=lambda x: -1 * (x[-6:-3] - x[-3:]),   # TODO: check by setting breakpoint in goalenv
        obs2goal_transform=lambda x: x[-3:],
        obs2start_transform=lambda x: x[-3:], #TODO, make sure transforms are correct!
        # start is just the initial lego position
        terminal_eps = v['terminal_eps'],  # TODO: potentially make more lenient?
        distance_metric = 'L2',
        extend_distance_rew = v['extend_distance_rew'],  # I think this turns off L2 distance reward
        # distance_rew = True, # check, I think this checks the desired distance
        terminate_env = v['terminate_env'],
    )

    # Follows Ignasi's code
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have n hidden layers, each with k hidden units.
        hidden_sizes=(64, 64, 64),
        init_std=0.1,
        output_gain=0.1,
        )

    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']
    outer_iter = 0
    logger.log("Generating the initial heat map...")
    plot_pushing(policy, env, report, bounds=(v['lego_init_lower'], v['lego_init_upper']),
                                                    center=v['lego_init_center'], itr=outer_iter)

    sigma = v['sigma']
    all_starts = StateCollection(distance_threshold=v['coll_eps'])
    # generate starting positions with Gaussian noise
    seed_starts = generate_starts_random(starts=[np.array(v['lego_target'][:2])], size = 1000, subsample=v['num_new_starts'], sigma = sigma,
                                            center=v['lego_target'][:2], range_lower=v['lego_init_lower'][:2], range_upper=v['lego_init_upper'][:2], noise="gaussian")

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")
        starts = generate_starts_random(starts=seed_starts, size=1000, subsample=v['num_new_starts'], sigma=sigma * 3,
                                        center=v['lego_init_center'][:2], range_lower=v['lego_init_lower'][:2], range_upper=v['lego_init_upper'][:2],
                                        noise="gaussian")
        # PLOT BEFORE TRAINING
        labels = label_states(starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'], key='goal_reached')
        plot_labeled_states(starts, labels, report=report, itr=outer_iter,
                            limit=v['lego_init_upper'][0],  # add limit in later
                            center=v['lego_init_center'][:2][::-1], maze_id=-1)

        if v['replay_buffer'] and outer_iter > 0 and all_starts.size > 0:
            old_starts = all_starts.sample(v['num_old_starts'])
            starts = np.vstack([starts, old_starts])

        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment start generator")
            env.update_start_generator(
                UniformListStateGenerator(
                    starts.tolist(), persistence=v['persistence'], with_replacement=v['with_replacement'],
                )
            )

            logger.log("Training the algorithm")
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=v['batch_size'],
                max_path_length=v['max_path_length'],  #100 #making path length longer is fine because of early termination
                n_itr=v['inner_iters'],
                # max_path_length=50,
                # batch_size=1000,
                # n_itr =2,
                discount=0.95,
                gae_lambda=0.98,
                step_size=0.01,
                # goal_generator=goal_generator,
                action_limiter=None,
                optimizer_args={'subsample_factor': 0.1},
                )

            algo.train()

        labels = label_states(starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'], key='goal_reached')
        plot_labeled_states(starts, labels, report=report, itr=outer_iter,
                            limit=v['lego_init_upper'][0] + 0.02,  # add limit in later
                            center=v['lego_init_center'][:2][::-1], maze_id=-1)

        logger.log("Generating heat map")
        plot_pushing(policy, env, report, bounds=(v['lego_init_lower'], v['lego_init_upper']),
                     center=v['lego_init_center'], itr=outer_iter)

        labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.dump_tabular(with_prefix=False)
        report.new_row()

        filtered_raw_starts = [start for start, label in zip(starts, labels) if label[0] == 1] # high reward starts
        if len(filtered_raw_starts) > 0:  # add a tone of noise if all the states I had ended up being high_reward!
            seed_starts = filtered_raw_starts
        else:
            seed_starts = generate_starts_random(starts=seed_starts, size=1000, subsample=v['num_new_starts'], sigma=sigma * 3,
                                        center=v['lego_init_center'][:2], range_lower=v['lego_init_lower'][:2], range_upper=v['lego_init_upper'][:2],
                                        noise="gaussian")
        logger.log("Number of good starts: {}".format(filtered_raw_starts))
        all_starts.append(filtered_raw_starts)




vg = VariantGenerator()
vg.add('seed', [1])
# vg.add('seed', [2,12,22,32,42])

# Environment parameters
vg.add('init_hand', [[0.6,  0.4,  0.5025],])
vg.add('lego_target', [(0.6, 0.15, 0.5025),])
vg.add('lego_init_center', [(0.6, 0.25, 0.5025),])
vg.add('lego_init_lower', [(-0.2, -0.1, 0),])
vg.add('lego_init_upper', [(0.2, 0.1, 0),])
#todo: plot target

# Optimizer parameters
vg.add('inner_iters', [5])
vg.add('outer_iters', [6000])
vg.add('batch_size', [15000])
vg.add('max_path_length', [70]) #TODO: make longer?

# Goal generation parameters
vg.add('terminal_eps', [0.03])
vg.add('extend_distance_rew', [False])
vg.add('terminate_env', [True])

# gan configs
vg.add('replay_buffer', [True])
vg.add('num_labels', [1])  # 1 for single label, 2 for high/low and 3 for learnability
vg.add('gan_generator_layers', [[200, 200]])
vg.add('gan_discriminator_layers', [[128, 128]])
vg.add('gan_noise_size', [5])
vg.add('start_noise_level', [0]) # no noise when sampling
vg.add('gan_outer_iters', [500])
vg.add('num_new_starts', [200])
vg.add('num_old_starts', [100])
vg.add('coll_eps', [0.03])
vg.add('horizon', [100])
vg.add('n_traj', [5])
vg.add('sigma', [0.02])
vg.add('random_angle', [False])

# start generator
vg.add('persistence', [1])
vg.add('with_replacement', [True])

for vv in vg.variants():
    # run_task(vv) # uncomment when debugging

    run_experiment_lite(
        stub_method_call=run_task,
        use_gpu=False,
        variant=vv,
        # Number of parallel workers for sampling
        # n_parallel=32,
        n_parallel=3, # use cpu_count in the future
        snapshot_mode="last",
        seed=vv['seed'],
        mode="local",
        # mode="ec2",
        exp_prefix="blockpushing1",
        # exp_name= "decaying-decaying-gamma" + str(t),
        # plot=True,
    )

