import os
import os.path as osp
import random

import matplotlib
import tensorflow as tf
import tflearn

matplotlib.use('Agg')

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import run_experiment_lite

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.box2d.pendulum_env import PendulumEnv
from rllab.misc.instrument import VariantGenerator

from sandbox.young_clgan.envs.base import GoalExplorationEnv
from sandbox.young_clgan.envs.base import UniformListGoalGenerator, FixedGoalGenerator,\
    UniformGoalGenerator, update_env_goal_generator
from sandbox.young_clgan.goal.evaluator import *
from sandbox.young_clgan.goal.generator import GoalGAN
# from sandbox.young_clgan.lib.goal.utils import *
from sandbox.young_clgan.logging.html_report import HTMLReport
from sandbox.young_clgan.logging.visualization import *
from sandbox.young_clgan.logging.logger import ExperimentLogger

from sandbox.young_clgan.utils import initialize_parallel_sampler
initialize_parallel_sampler()

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]

if __name__ == '__main__':

    exp_prefix = 'goalGAN-pendulum-debug'
    vg = VariantGenerator()
    vg.add('seed', range(30, 40, 10))
    # # GeneratorEnv params
    vg.add('goal_range', [np.pi])
    vg.add('goal', [(np.pi, 0)])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('angle_idxs', [((0, 1),)]) # these are the idx of the obs corresponding to angles (here the first 2)
    vg.add('distance_metric', ['L2'])
    vg.add('terminal_bonus', [1e3])
    vg.add('terminal_eps', [0.1])
    vg.add('goal_weight', [0])
    vg.add('goal_reward', ['NegativeDistance'])
    # old hyperparams
    vg.add('outer_iters', [50])
    vg.add('inner_iters', [5])
    vg.add('horizon', [200])
    vg.add('pg_batch_size', [20000])
    #############################################
    vg.add('min_reward', [0.1])  # now running it with only the terminal reward of 1!
    vg.add('max_reward', [1e3])
    vg.add('improvement_threshold', [0.1])  # is this based on the reward, now discounted success rate --> push for fast
    # gan_configs
    vg.add('goal_noise_level', [1])  # ???
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
        inner_env = normalize(PendulumEnv())
        # inner_env.wrapped_env.reset(noisy=False)
        center = [0, 0]
        goal_size = np.size(v['goal'])
        fixed_goal_generator = FixedGoalGenerator(goal=v['goal'])
        uniform_goal_generator = UniformGoalGenerator(goal_size=np.size(v['goal']), bounds=v['goal_range'], center=center)

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

        env = GoalExplorationEnv(env=inner_env, goal_generator=fixed_goal_generator, goal_reward=v['goal_reward'],
                                 distance_metric=v['distance_metric'],
                                 goal_weight=v['goal_weight'], terminal_bonus=v['terminal_bonus'],
                                 angle_idxs=(0,))  # this goal_generator will be updated by a uniform after

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            # Fix the variance since different goals will require different variances, making this parameter hard to learn.
            learn_std=False
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
        logger.log("Initializing report and plot_policy_reward...")
        log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
        report = HTMLReport(osp.join(log_dir, 'report.html'))
        report.add_header("{}".format(EXPERIMENT_TYPE))
        # report.add_text(format_dict(vg.variant))

        # img = plot_policy_reward(
        #     policy, env, v['goal_range'],
        #     horizon=v['horizon'], grid_size=5,
        #     fname='{}/policy_reward_init.png'.format(log_dir),
        # )
        # report.add_image(img, 'policy performance initialization\n')

        # Pretrain GAN with uniform distribution on the GAN output space and log a sample
        logger.log("pretraining the GAN with uniform...")
        gan.pretrain_uniform()
        # img = plot_gan_samples(gan, v['goal_range'], '{}/start.png'.format(log_dir))
        # report.add_image(img, 'GAN pretrained uniform')

        report.save()
        report.new_row()

        all_goals = np.zeros((0, goal_size))

        logger.log("Starting the outer iterations")
        for outer_iter in range(v['outer_iters']):

            # Train GAN
            logger.log("Sampling goals...")
            raw_goals, _ = gan.sample_goals_with_noise(2000)

            if outer_iter > 0:
                # sampler uniformly 2000 old goals and add them to the training pool (50/50)
                old_goal_indices = np.random.randint(0, all_goals.shape[0], 2000)
                old_goals = all_goals[old_goal_indices, :]
                goals = np.vstack([raw_goals, old_goals])
            else:
                goals = raw_goals

            # append new goals to list of all goals (replay buffer)
            all_goals = np.vstack([all_goals, raw_goals])

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

            logger.log("Plot performance policy on full grid...")
            img = plot_policy_reward(
                policy, env, v['goal_range'],
                horizon=v['horizon'],
                max_reward=v['max_reward'],
                grid_size=10,
                # fname='{}/policy_reward_{}.png'.format(log_config.plot_dir, outer_iter),
            )
            report.add_image(img, 'policy performance\n itr: {}'.format(outer_iter))
            report.save()

            # this re-evaluate the final policy in the collection of goals
            logger.log("Generating labels by re-evaluating policy on List of goals...")
            labels = label_goals(
                goals, env, policy, v['horizon'],
                min_reward=v['min_reward'],
                max_reward=v['max_reward'],
                old_rewards=rewards_before,
                improvement_threshold=v['improvement_threshold']
            )
            img = plot_labeled_samples(
                samples=goals, labels=labels, limit=v['goal_range'] + 5,
                # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
            )
            report.add_image(img, 'goals\n itr: {}'.format(outer_iter), width=500)
            report.save()
            report.new_row()

            logger.log("Training GAN...")
            gan.train(
                goals, labels,
                v['gan_outer_iters'],
                v['gan_generator_iters'],
                v['gan_discriminator_iters']
            )


            logger.log("Evaluating performance on Unif and Fix Goal Gen...")
            # log some more on how the pendulum performs the upright and general task
            with logger.tabular_prefix('UnifGoalGen_'):
                update_env_goal_generator(env, goal_generator=uniform_goal_generator)
                evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=10)
            with logger.tabular_prefix('FixGoalGen_'):
                update_env_goal_generator(env, goal_generator=fixed_goal_generator)
                evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5)
            logger.dump_tabular(with_prefix=False)


    for vv in vg.variants():
        run_experiment_lite(
            run_task,
            pre_commands=[
                          'export MPLBACKEND=Agg',
                          'pip install --upgrade pip',
                          'pip install --upgrade -I tensorflow',
                          'pip install git+https://github.com/tflearn/tflearn.git',
                          'pip install dominate',
                          'pip install scikit-image',
                          'conda install numpy -n rllab3 -y',
                          ],
            variant=vv,
            mode='ec2',
            sync_s3_html=True,
            # n_parallel=n_parallel,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=vv['seed'],
            exp_prefix=exp_prefix,

        )
