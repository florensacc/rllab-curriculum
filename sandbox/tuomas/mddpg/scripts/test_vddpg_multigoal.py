import tensorflow as tf
import numpy as np
import os

from rllab import config

from rllab.envs.proxy_env import ProxyEnv

from rllab.misc import logger

from sandbox.tuomas.mddpg.policies.stochastic_policy \
    import DummyExplorationStrategy

from rllab.exploration_strategies.ou_strategy import OUStrategy

from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('actor_lr', 0.001, 'Base learning rate for actor.')
flags.DEFINE_float('critic_lr', 0.001, 'Base learning rate for critic.')
flags.DEFINE_integer('path_length', 100, 'Maximum path length.')
flags.DEFINE_integer('n_particles', 64, 'Number of particles.')
flags.DEFINE_string('alg', 'vddpg', 'Algorithm.')
flags.DEFINE_string('policy', 'stochastic',
                    'Policy (DETERMINISTIC/stochastic')
flags.DEFINE_string('output', 'exp00', 'Experiment name.')

# flags.DEFINE_string('snapshot_dir', None, 'Snapshot directory.')

#if FLAGS.snapshot_dir is not None:
#    logger.set_snapshot_dir(FLAGS.snapshot_dir)
tabular_log_file = os.path.join(config.LOG_DIR, 'multireward',
                                FLAGS.output + '.txt')
snapshot_dir = os.path.join(config.LOG_DIR, 'multireward', FLAGS.output)
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(snapshot_dir)

#flags.DEFINE_string('save_path', '', 'Path where the plots are saved.')

if FLAGS.alg == 'ddpg':
    from sandbox.tuomas.mddpg.algos.ddpg import DDPG as Alg
elif FLAGS.alg == 'vddpg':
    from sandbox.tuomas.mddpg.algos.vddpg import VDDPG as Alg


if FLAGS.policy == 'deterministic':
    from sandbox.tuomas.mddpg.policies.nn_policy \
        import FeedForwardPolicy as Policy
elif FLAGS.policy == 'stochastic':
    from sandbox.tuomas.mddpg.policies.stochastic_policy \
        import StochasticNNPolicy as Policy


class AlgTest(Alg):

    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(AlgTest, self).__init__(*args, **kwargs)

        self._base_env = self.env
        while isinstance(self._base_env, ProxyEnv):
            self._base_env = self._base_env.wrapped_env

        # Plot settings.
        self._lim_action = 2.  # Limits for plotting.
        self._n_training_paths = 10
        self._n_test_paths = 10

        self._training_path_counter = 0
        self._test_path_counter = 0

        # Evaluate the Q function for the following states.

        self._q_obs_list = np.array([[-2.5, 0.0],
                                     [0.0, 0.0],
                                     [2.5, 2.5]])
        self._q_list = [self.qf]

        self._current_training_path = []
        self._h_training_paths = []
        self._h_test_paths = []

        self._init_plots()


    @overrides
    def process_env_info(self, info, flush):
        self._current_training_path.append(info)
        if flush:
            self._h_training_paths.append(
                self._base_env.plot_path(self._current_training_path,
                                         self._ax_env, 'r')
            )
            plt.draw()
            plt.pause(0.001)
            self._current_training_path = []

        if len(self._h_training_paths) > self._n_training_paths:
            h = self._h_training_paths.pop(0)
            h.remove()

    def _init_plots(self):

        # Set critic figure and all axes up.
        self._critic_fig = plt.figure(figsize=(18, 7))
        self._ax_critics = []
        n_plots = self._q_obs_list.shape[0] * len(self._q_list)
        for i in range(n_plots):
            ax = self._critic_fig.add_subplot(100 + n_plots * 10 + i + 1)
            self._ax_critics.append(ax)
            plt.axis('equal')
            ax.set_xlim((-self._lim_action, self._lim_action))
            ax.set_ylim((-self._lim_action, self._lim_action))

        # Set environment plot up
        self._env_fig = plt.figure(figsize=(7, 7))
        self._ax_env = self._env_fig.add_subplot(111)
        self._base_env.set_axis(self._ax_env)

    def _eval_critic(self, o, q):
        xx = np.arange(-self._lim_action, self._lim_action, 0.05)
        X, Y = np.meshgrid(xx, xx)
        all_actions = np.vstack([X.flatten(), Y.flatten()]).transpose()
        obs = np.array([o] * all_actions.shape[0])

        feed = q.get_feed_dict(obs, all_actions)

        Q = self.sess.run(q.output, feed).reshape(X.shape)
        return X, Y, Q

    @overrides
    def evaluate(self, epoch, es_path_returns):
        # Plot critic and samples.

        itr = 0
        for obs in self._q_obs_list:
            for q in self._q_list:
                ax = self._ax_critics[itr]
                X, Y, Q = self._eval_critic(obs, q)

                ax.clear()
                cs = ax.contour(X, Y, Q, 20)
                ax.clabel(cs, inline=1, fontsize=10, fmt='%.0f')

                # sample and plot actions
                N = FLAGS.n_particles
                all_obs = np.array([obs] * N)
                all_actions = self.policy.get_actions(all_obs)[0]

                x = all_actions[:, 0]
                y = all_actions[:, 1]
                ax.plot(x, y, '*')

                itr += 1

        # Remove old test paths
        while len(self._h_test_paths):
            self._h_test_paths.pop().remove()

        # Do test rollouts and plot.
        env_state = self._base_env.__getstate__()
        successes = 0.
        for _ in range(self._n_test_paths):
            o = self.env.reset()
            info_list = []
            rewards = []
            for t in range(FLAGS.path_length):
                a, _ = self.policy.get_action(o)
                o, r, d, info = self.env.step(a)
                info_list.append(info)
                rewards.append(r)
                if d:
                    successes += 1.
                    break
            self._h_test_paths.append(
                self._base_env.plot_path(info_list, self._ax_env)
            )
        self._base_env.__setstate__(env_state)
        #mean_rewards = np.mean(np.stack(rewards, axis=1), axis=1)

        plt.draw()
        plt.pause(0.001)

        logger.record_tabular("success_rate", successes / self._n_test_paths)
        #logger.record_tabular("avg test int reward", mean_rewards[0])
        #logger.record_tabular("avg test ext reward", mean_rewards[1])

        # TODO: hacky way to check if this is VDDPG instance without loading
        # VDDPG class.
        if hasattr(self, 'alpha'):
            if epoch % 50 == 0:
                self.alpha /= 3.


# -------------------------------------------------------------------
def test():
    from sandbox.tuomas.mddpg.kernels.gaussian_kernel import \
        SimpleAdaptiveDiagonalGaussianKernel
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.tuomas.mddpg.envs.multi_goal_env import MultiGoalEnv
    from sandbox.tuomas.mddpg.critics.nn_qfunction \
        import FeedForwardCritic, MultiCritic


    alg_kwargs = dict(
        epoch_length=100,  # evaluate / plot per SVGD step
        min_pool_size=1000,#1000,  # must be at least 2
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
        scale_reward=0.1,
    )

    policy_kwargs = dict(
    )

    if FLAGS.policy == 'deterministic':
        policy_kwargs.update(dict(
            observation_hidden_sizes=(100, 100)
        ))

    elif FLAGS.policy == 'stochastic':
        policy_kwargs.update(dict(
            sample_dim=2,
            freeze_samples=False,
            K=FLAGS.n_particles,
            output_nonlinearity=tf.tanh,
            hidden_dims=(100, 100),
            W_initializer=None,
            output_scale=2.0,
        ))

    if FLAGS.alg == 'vddpg':
        alg_kwargs.update(dict(
            train_actor=True,
            train_critic=True,
            q_target_type='max',
            K=FLAGS.n_particles,
            alpha=10.,
        ))

    # ----------------------------------------------------------------------
    env = TfEnv(MultiGoalEnv())

    #es = DummyExplorationStrategy()
    es = OUStrategy(env_spec=env.spec, mu=0, theta=0.15, sigma=0.3)

    qf = FeedForwardCritic(
        "critic1",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        observation_hidden_sizes=(),
        embedded_hidden_sizes=(100, 100),
    )

    policy = Policy(
        scope_name='actor',
        observation_dim=env.observation_space.flat_dim,
        action_dim=env.action_space.flat_dim,
        **policy_kwargs,
    )

    kernel = SimpleAdaptiveDiagonalGaussianKernel(
        "kernel",
        dim=env.action_space.flat_dim,
    )

    if FLAGS.alg == 'vddpg':
       alg_kwargs.update(dict(
            kernel=kernel,
            q_prior=None,
       ))

    algorithm = AlgTest(
        env=env,
        exploration_strategy=es,
        policy=policy,
        eval_policy=None,
        qf=qf,
        **alg_kwargs
    )
    algorithm.train()

if __name__ == "__main__":
    test() # this way variables defined in test() are not global, hence
    # not accessible to other methods, avoiding bugs
