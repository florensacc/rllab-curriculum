import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import os.path as osp
import sys
import time
import random
import numpy as np
import scipy
import tensorflow as tf
import tflearn
from collections import OrderedDict

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import run_experiment_lite
from rllab.misc import logger

from sandbox.carlos_snn.envs.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.box2d.pendulum_env import PendulumEnv
from rllab.misc.instrument import VariantGenerator

from sandbox.young_clgan.lib.envs.base import GoalExplorationEnv, GoalIdxExplorationEnv
from sandbox.young_clgan.lib.envs.base import UniformListGoalGenerator, FixedGoalGenerator, UniformGoalGenerator, \
     update_env_goal_generator, generate_initial_goals

from sandbox.young_clgan.lib.goal.evaluator import *
from sandbox.young_clgan.lib.goal.generator import *
from sandbox.young_clgan.lib.goal.utils import *
from sandbox.young_clgan.lib.logging.html_report import format_dict, HTMLReport
from sandbox.young_clgan.lib.logging.visualization import *
from sandbox.young_clgan.lib.logging.logger import ExperimentLogger

# from sandbox.young_clgan.lib.utils import initialize_parallel_sampler
#
# initialize_parallel_sampler()

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]

if __name__ == '__main__':

    exp_prefix = 'goalGAN-point'
    vg = VariantGenerator()
    vg.add('seed', range(30, 40, 10))
    # # GeneratorEnv params
    vg.add('goal', [(0, 0, 0)])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('goal_range', [2, 4])  # this will be used also as bound of the state_space
    vg.add('state_bounds', lambda goal_range: [(0.5, ) + (goal_range,) * 7])
    # vg.add('angle_idxs', [((0, 1),)]) # these are the idx of the obs corresponding to angles (here the first 2)
    vg.add('reward_dist_threshold', [0.2, 0.5])
    vg.add('distance_metric', ['L2'])
    vg.add('terminal_bonus', [0])
    vg.add('terminal_eps', [0.5])  # if hte terminal bonus is 0 it doesn't kill it! Just count how many reached center
    vg.add('goal_weight', [1])
    # vg.add('goal_reward', ['NegativeDistance'])
    #############################################
    vg.add('min_reward', [0.1])  # now running it with only the terminal reward of 1!
    vg.add('max_reward', [1e3])
    vg.add('improvement_threshold', [10])  # is this based on the reward, now discounted success rate --> push for fast
    # old hyperparams
    vg.add('num_new_goals', [200])
    vg.add('num_old_goals', [200])
    vg.add('coll_eps', lambda reward_dist_threshold: [reward_dist_threshold/4, reward_dist_threshold])
    vg.add('outer_iters', [200])
    vg.add('inner_iters', [5])
    vg.add('horizon', [200])
    vg.add('pg_batch_size', [20000])
    # gan_configs
    vg.add('goal_noise_level', [0.1])  # ???
    vg.add('gan_outer_iters', [5])
    vg.add('gan_discriminator_iters', [200])
    vg.add('gan_generator_iters', [5])
    vg.add('GAN_batch_size', [128])  # proble with repeated name!!
    vg.add('GAN_generator_activation', ['relu'])
    vg.add('GAN_discriminator_activation', ['relu'])
    vg.add('GAN_generator_optimizer', [tf.train.AdamOptimizer])
    vg.add('GAN_generator_optimizer_stepSize', [0.001])
    vg.add('GAN_discriminator_optimizer', [tf.train.AdamOptimizer])
    vg.add('GAN_discriminator_optimizer_stepSize', [0.001])
    vg.add('GAN_generator_weight_initializer', [tflearn.initializations.truncated_normal])
    vg.add('GAN_generator_weight_initializer_stddev', [0.05])
    vg.add('GAN_discriminator_weight_initializer', [tflearn.initializations.truncated_normal])
    vg.add('GAN_discriminator_weight_initializer_stddev', [0.02])
    vg.add('GAN_discriminator_batch_noise_stddev', [1e-2])


    def run_task(v):
        random.seed(v['seed'])
        np.random.seed(v['seed'])

        tf_session = tf.Session()

        # goal generators
        logger.log("Initializing the goal generators and the inner env...")
        goal_dim = np.size(v['goal'])
        inner_env = normalize(PointEnv(dim=goal_dim, state_bounds=v['state_bounds']))
        # inner_env = normalize(PendulumEnv())

        center = np.zeros(np.array(v['goal']).size)
        goal_size = np.size(v['goal'])
        fixed_goal_generator = FixedGoalGenerator(goal=v['goal'])
        uniform_goal_generator = UniformGoalGenerator(goal_size=goal_dim, bounds=v['goal_range'],
                                                      center=center)
        feasible_goal_ub = np.array(v['state_bounds'])[:goal_dim]
        uniform_feasible_goal_generator = UniformGoalGenerator(goal_size=goal_dim, bounds=[-1 * feasible_goal_ub,
                                                                                           feasible_goal_ub])

        # GAN
        logger.log("Instantiating the GAN...")
        gan_configs = {key[4:]: value for key, value in v.items() if 'GAN_' in key}
        for key, value in gan_configs.items():
            if value is tf.train.AdamOptimizer:
                gan_configs[key] = tf.train.AdamOptimizer(gan_configs[key + '_stepSize'])
            if value is tflearn.initializations.truncated_normal:
                gan_configs[key] = tflearn.initializations.truncated_normal(stddev=gan_configs[key + '_stddev'])

        gan = GoalGAN(
            goal_size=goal_size,
            evaluater_size=3,
            goal_range=v['goal_range'],
            goal_noise_level=v['goal_noise_level'],
            generator_layers=[256, 256],
            discriminator_layers=[128, 128],
            noise_size=4,
            tf_session=tf_session,
            configs=gan_configs,
        )
        env = GoalIdxExplorationEnv(env=inner_env, goal_generator=fixed_goal_generator,
                                    idx=np.arange(np.size(v['goal'])),
                                    reward_dist_threshold=v['reward_dist_threshold'],
                                    distance_metric=v['distance_metric'], goal_weight=v['goal_weight'],
                                    terminal_eps=v['terminal_eps'], terminal_bonus=v['terminal_bonus'],
                                    )  # this goal_generator will be updated by a uniform after

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            # Fix the variance since different goals will require different variances, making this parameter hard to learn.
            learn_std=False,
            # output_gain=0.1,
            init_std=0.1,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        logger.log("Initializing report and plot_policy_reward...")
        log_dir = logger.get_snapshot_dir()
        report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)
        report.add_header("{}".format(EXPERIMENT_TYPE))
        report.add_text(format_dict(v))

        # img = plot_policy_reward(
        #     policy, env, v['goal_range'],
        #     horizon=v['horizon'], grid_size=5,
        #     fname='{}/policy_reward_init.png'.format(log_dir),
        # )
        # report.add_image(img, 'policy performance initialization\n')

        logger.log("pretraining the GAN...")
        # gan.pretrain_uniform()
        gan.pretrain(
            generate_initial_goals(env, policy, v['goal_range'], horizon=v['horizon'])
        )
        img = plot_gan_samples(gan, v['goal_range'], '{}/start.png'.format(log_dir))
        report.add_image(img, 'GAN pretrained uniform')

        report.save()
        report.new_row()

        all_goals = GoalCollection(0.5)

        all_mean_rewards = []
        all_coverage = []

        logger.log("Starting the outer iterations")
        for outer_iter in range(v['outer_iters']):

            # Train GAN
            logger.log("Sampling goals...")
            raw_goals, _ = gan.sample_goals_with_noise(200)

            if outer_iter > 0:
                # sampler uniformly 2000 old goals and add them to the training pool (50/50)
                old_goals = all_goals.sample(num_old_goals)
                goals = np.vstack([raw_goals, old_goals])
            else:
                goals = raw_goals

            # append new goals to list of all goals (replay buffer)
            all_goals.append(raw_goals)

            logger.log("Evaluating goals before inner training...")
            rewards_before = evaluate_goals(goals, env, policy, v['horizon'])

            logger.log("Perform TRPO with UniformListGoalGenerator...")
            with ExperimentLogger(log_dir, outer_iter, snapshot_mode='last', hold_outter_log=True):
                # set goal generator to uniformly sample from selected all_goals
                update_env_goal_generator(
                    env,
                    UniformListGoalGenerator(
                        goals.tolist()
                    )
                )

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

            # logger.log("Plot performance policy on full grid...")
            # img = plot_policy_reward(
            #     policy, env, v['goal_range'],
            #     horizon=v['horizon'],
            #     max_reward=v['max_reward'],
            #     grid_size=10,
            #     # fname='{}/policy_reward_{}.png'.format(log_config.plot_dir, outer_iter),
            # )
            # report.add_image(img, 'policy performance\n itr: {}'.format(outer_iter))
            # report.save()

            # this re-evaluate the final policy in the collection of goals
            logger.log("Generating labels by re-evaluating policy on List of goals...")
            labels = label_goals(
                goals, env, policy, v['horizon'],
                min_reward=v['min_reward'],
                max_reward=v['max_reward'],
                old_rewards=rewards_before,
                improvement_threshold=v['improvement_threshold']
            )
            goal_classes, text_labels = convert_label(labels)
            total_goals = labels.shape[0]
            goal_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
            for k in text_labels.keys():
                frac = np.sum(goal_classes == k) / total_goals
                logger.record_tabular('GenGoal_frac_' + text_labels[k], frac)
                goal_class_frac[text_labels[k]] = frac

            img = plot_labeled_samples(
                samples=goals, sample_classes=goal_classes, text_labels=text_labels, limit=v['goal_range'],
                # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
            )
            summary_string = ''
            for key, value in goal_class_frac.items():
                summary_string += key + ' frac: ' + str(value) + '\n'
            report.add_image(img, 'itr: {}\nLabels of generated goals:\n{}'.format(outer_iter, summary_string), width=500)

            # log feasibility of generated goals
            feasible = np.array([1 if env.feasible_goal_space.contains(goal) else 0 for goal in goals], dtype=int)
            feasibility_rate = np.mean(feasible)
            logger.record_tabular('GenGoalFeasibilityRate', feasibility_rate)
            img = plot_labeled_samples(
                samples=goals, sample_classes=feasible, text_labels={0: 'Infeasible', 1: "Feasible"},
                markers={0: 'v', 1: 'o'}, limit=v['goal_range'],
                # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
            )
            report.add_image(img, 'feasibility of generated goals: {}\n itr: {}'.format(feasibility_rate, outer_iter),
                             width=500)

            logger.log("Training GAN...")
            gan.train(
                goals, labels,
                v['gan_outer_iters'],
                v['gan_generator_iters'],
                v['gan_discriminator_iters']
            )

            # log some more on how the pendulum performs the upright and general task
            logger.log("Evaluating performance on Unif and Fix Goal Gen...")
            with logger.tabular_prefix('UnifFeasGoalGen_'):
                update_env_goal_generator(env, goal_generator=uniform_feasible_goal_generator)
                evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=20, fig_prefix='UnifFeasGoalGen_',
                                  report=report)
            # with logger.tabular_prefix('FixGoalGen_'):
            #     update_env_goal_generator(env, goal_generator=fixed_goal_generator)
            #     evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5, fig_prefix='FixGoalGen',
            #                       report=report)
            logger.dump_tabular(with_prefix=False)

            report.save()
            report.new_row()


    for vv in vg.variants():
        run_experiment_lite(
            run_task,
            pre_commands=['export MPLBACKEND=Agg',
                          'pip install --upgrade pip',
                          'pip install --upgrade -I tensorflow',
                          'pip install git+https://github.com/tflearn/tflearn.git',
                          'pip install dominate',
                          'pip install scikit-image',
                          'conda install numpy -n rllab3 -y',
                          ],
            sync_s3_html=True,
            variant=vv,
            mode='local',
            # n_parallel=n_parallel,
            n_parallel=2,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=vv['seed'],
            exp_prefix=exp_prefix,
        )
