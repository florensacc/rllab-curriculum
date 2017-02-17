import tensorflow as tf
import numpy as np
import pickle
from matplotlib.backends.backend_pdf import PdfPages

from rllab.envs.proxy_env import ProxyEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy


from sandbox.tuomas.mddpg.misc.rollout import rollout
from sandbox.tuomas.mddpg.policies.stochastic_policy import \
    DummyExplorationStrategy

from sandbox.tuomas.mddpg.critics.gaussian_critic import MixtureGaussian2DCritic

#from sandbox.tuomas.mddpg.algos.vddpg import VDDPG
from rllab.misc.overrides import overrides
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('actor_lr', 0.01, 'Base learning rate for actor.')
flags.DEFINE_float('critic_lr', 0.001, 'Base learning rate for critic.')
flags.DEFINE_integer('path_length', 100, 'Maximum path length.')
flags.DEFINE_integer('n_particles', 32, 'Number of particles.')
flags.DEFINE_string('alg', 'vddpg', 'Algorithm.')
flags.DEFINE_string('policy', 'stochastic',
                    'Policy (DETERMINISTIC/stochastic')
flags.DEFINE_string('save_path', '', 'Path where the plots are saved.')

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
    lim = 2.
    plots_initialized = False
    eval_counter = 0
    n_paths = 5

    def eval_critic(self, o):
        xx = np.arange(-self.lim, self.lim, 0.05)
        X, Y = np.meshgrid(xx, xx)
        all_actions = np.vstack([X.ravel(), Y.ravel()]).transpose()
        obs = np.array([o] * all_actions.shape[0])

        feed = {
            self.qf.observations_placeholder: obs,
            self.qf.actions_placeholder: all_actions
        }
        Q = self.sess.run(self.qf.output, feed).reshape(X.shape)
        return X, Y, Q

    def obs_to_pos(self, obs):
        cos1 = obs[:, 0]
        cos2 = obs[:, 1]
        sin1 = obs[:, 2]
        sin2 = obs[:, 3]

        joint1_len = 0.1
        joint2_len = 0.11  # This is actually fingertip position.

        # joint 1 location
        x1 = joint1_len * cos1
        y1 = joint1_len * sin1

        # joint 2 location
        x2 = x1 + joint2_len * (cos1 * cos2 - sin1 * sin2)
        y2 = y1 + joint2_len * (sin1 * cos2 + cos1 * sin2)

        xy = np.stack((x2, y2), axis=1)
        return xy

    @overrides
    def evaluate(self, epoch, es_path_returns):
        self.eval_counter += 1

        # Evaluate the Q function for the following states.
        qpos_list = np.array([[0.0, 0.0],
                              [np.pi / 8., 0.0]])

        env = self.env
        while isinstance(env, ProxyEnv):
            env = env.wrapped_env


        if not self.plots_initialized:
            # Set up all critic plots.
            self._critic_fig = plt.figure(figsize=(20, 7))
            self.ax_critics = []
            n_plots = qpos_list.shape[0]
            for i in range(n_plots):
                ax = self._critic_fig.add_subplot(100 + n_plots * 10 + i + 1)
                self.ax_critics.append(ax)
                plt.axis('equal')
                ax.set_xlim((-self.lim, self.lim))
                ax.set_ylim((-self.lim, self.lim))

            # Setup actor plots
            fig = plt.figure(figsize=(7, 7))
            self.ax_paths = fig.add_subplot(111)
            env.plot_env(self.ax_paths)
            plt.axis('equal')
            self.ax_paths.grid(True)
            self.ax_paths.set_xlim((-0.3, 0.3))
            self.ax_paths.set_ylim((-0.3, 0.3))

            self.h_training_paths = []
            self.h_test_paths = []
            for i in range(self.n_paths):
                self.h_training_paths += self.ax_paths.plot([], [], 'r')
                self.h_test_paths += self.ax_paths.plot([], [], 'b')

            self.plots_initialized = True


        ## Do rollouts.
        for line in self.h_test_paths:
            #self.policy.reset()
            path = rollout(self.env, self.policy, FLAGS.path_length)
            xy = self.obs_to_pos(path['observations'])
            line.set_data(np.transpose(xy))

        # DEBUG: use training rollouts
        if len(self.db.paths) >= self.n_paths:
            for i, line in enumerate(self.h_training_paths):
                xy = self.obs_to_pos(self.db.paths[-i-1][0])
                line.set_data(np.transpose(xy))

        plt.draw()
        plt.pause(0.001)

        for ax_critic, qpos in zip(self.ax_critics, qpos_list):
            obs = env.reset_mujoco(qpos_init=qpos)

            X, Y, Q = self.eval_critic(obs)

            ax_critic.clear()
            cs = ax_critic.contour(X, Y, Q, 20)

            ax_critic.clabel(cs, inline=1, fontsize=10, fmt='%.0f')

            # sample and plot actions
            N = FLAGS.n_particles
            all_obs = np.array([obs] * N)
            all_actions = self.policy.get_actions(all_obs)[0]
            #print(all_actions)h_test_paths

            x = all_actions[:, 0]
            y = all_actions[:, 1]
            ax_critic.plot(x, y, '*')

        plt.draw()
        plt.pause(0.001)

        # Do one rollout.
        #return # Don't do

        #o = env.reset_mujoco()
        #for _ in range(100):
        #    a, _ = self.policy.get_action(o)
        #    o, r, d, _ = self.env.step(a)
        #    self.env.render()
        #    if d:
        #        break
        #    plt.pause(0.001)

        #    import pdb; pdb.set_trace()

            #print(a)

        ## TODO: hacky way to check if this is VDDPG instance without loading
        ## VDDPG class.
        #if hasattr(self, 'alpha'):
        #    if self.eval_counter % 50 == 0:
        #        self.alpha /= 3

# -------------------------------------------------------------------
def test():
    from sandbox.tuomas.mddpg.kernels.gaussian_kernel import \
        SimpleAdaptiveDiagonalGaussianKernel
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.tuomas.mddpg.envs.multi_goal_reacher_env import \
        MultiGoalReacherEnv
    #from sandbox.tuomas.mddpg.envs.multi_goal_env import MultiGoalEnv
    #from sandbox.tuomas.mddpg.envs.multi_goal_env import MultiGoalEnv
    from sandbox.tuomas.mddpg.critics.nn_qfunction import FeedForwardCritic


    alg_kwargs = dict(
        epoch_length=100,  # evaluate / plot per SVGD step
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
        scale_reward=1.
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
            alpha=1.,
        ))

    # ----------------------------------------------------------------------
    env = TfEnv(MultiGoalReacherEnv())

    #es = DummyExplorationStrategy()
    es = OUStrategy(env_spec=env.spec, mu=0, theta=0.15, sigma=0.3)

    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        observation_hidden_sizes=(),
        embedded_hidden_sizes=(100, 100),
    )

    q_prior = MixtureGaussian2DCritic(
        "critic",
        observation_dim=env.observation_space.flat_dim,
        action_input=None,
        observation_input=None,
        weights=[1.],
        mus=[np.array((0., 0.))],
        sigmas=[1.0]
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
            q_prior=q_prior,
        ))

    algorithm = AlgTest(
        env=env,
        exploration_strategy=es,
        policy=policy,
        qf=qf,
        **alg_kwargs
    )
    algorithm.train()

if __name__ == "__main__":
    test() # this way variables defined in test() are not global, hence
    # not accessible to other methods, avoiding bugs
