import tensorflow as tf
import numpy as np
import os

from rllab import config
from rllab.misc import logger

from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.tuomas.mddpg.kernels.gaussian_kernel import \
    SimpleAdaptiveDiagonalGaussianKernel
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.tuomas.mddpg.envs.ant_env import AntEnv
from sandbox.tuomas.mddpg.critics.nn_qfunction import FeedForwardCritic
from sandbox.tuomas.mddpg.policies.stochastic_policy import \
    DummyExplorationStrategy, StochasticPolicyMaximizer
from sandbox.tuomas.mddpg.algos.vddpg import VDDPG
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('actor_lr', 0.01, 'Base learning rate for actor.')
flags.DEFINE_float('critic_lr', 0.001, 'Base learning rate for critic.')
flags.DEFINE_integer('path_length', 500, 'Maximum path length.')
flags.DEFINE_integer('n_particles', 32, 'Number of particles.')
flags.DEFINE_string('alg', 'vddpg', 'Algorithm.')
flags.DEFINE_string('save_path', '', 'Path where the plots are saved.')
flags.DEFINE_string('output', 'default', 'Experiment name.')


tabular_log_file = os.path.join(config.LOG_DIR, 'ant',
                                FLAGS.output, 'eval.log')
snapshot_dir = os.path.join(config.LOG_DIR, 'ant', FLAGS.output)
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(snapshot_dir)

raise NotImplementedError

def test():

    env = TfEnv(normalize(AntEnv(), clip=False))

    vddpg_kwargs = dict(
        epoch_length=1 * FLAGS.path_length,  # evaluate / plot per SVGD step
        min_pool_size=1000,  # must be at least 2
        #replay_pool_size=2,  # should only sample from recent experiences,
                        # note that the sample drawn can be one iteration older
        eval_samples=1,  # doesn't matter since we override evaluate()
        n_epochs=100000,  # number of SVGD steps
        policy_learning_rate=FLAGS.actor_lr,  # note: this is higher than DDPG's 1e-4
        qf_learning_rate=FLAGS.critic_lr,
        Q_weight_decay=0.00,
        #soft_target_tau=1.0
        max_path_length=FLAGS.path_length,
        batch_size=64,  # only need recent samples, though it's slow
        scale_reward=1.,
        train_actor=True,
        train_critic=True,
        q_target_type='soft',
        K=FLAGS.n_particles,
        alpha=1.,
        n_eval_paths=5,
        critic_train_frequency=1,
        actor_train_frequency=1,
        update_target_frequency=5000,
    )

    policy_kwargs = dict(
        sample_dim=2,
        freeze_samples=False,
        K=FLAGS.n_particles,
        output_nonlinearity=tf.tanh,
        hidden_dims=(100, 100),
        W_initializer=None,
        output_scale=2.0,
    )

    q_plot_settings = dict(
        xlim=(-2, 2),
        ylim=(-2, 2),
        obs_lst=(np.zeros(env.observation_space.flat_dim),),
        action_dims=(0, 1),
        axis_lims=((-2, 2), (-2, 2)),
    )

    env_plot_settings = dict(
        xlim=(-5, 5),
        ylim=(-5, 5),
    )

    # ----------------------------------------------------------------------

    #es = DummyExplorationStrategy()
    es = OUStrategy(env_spec=env.spec, mu=0, theta=0.15, sigma=0.03, clip=False)
    #es = OUStrategy(env_spec=env.spec, mu=0, theta=0.15, sigma=0.3, clip=False)

    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        observation_hidden_sizes=(),
        embedded_hidden_sizes=(100, 100),
    )

    policy = StochasticNNPolicy(
        scope_name='actor',
        observation_dim=env.observation_space.flat_dim,
        action_dim=env.action_space.flat_dim,
        **policy_kwargs,
    )

    eval_policy = StochasticPolicyMaximizer(
        N=100,
        actor=policy,
        critic=qf,
    )

    kernel = SimpleAdaptiveDiagonalGaussianKernel(
        "kernel",
        dim=env.action_space.flat_dim,
    )

    algorithm = VDDPG(
        env=env,
        exploration_strategy=es,
        policy=policy,
        eval_policy=eval_policy,
        qf=qf,
        q_plot_settings=q_plot_settings,
        env_plot_settings=env_plot_settings,
        kernel=kernel,
        **vddpg_kwargs
    )
    algorithm.train()

if __name__ == "__main__":
    test()  # this way variables defined in test() are not global, hence
    # not accessible to other methods, avoiding bugs
