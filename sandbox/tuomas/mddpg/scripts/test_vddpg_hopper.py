import tensorflow as tf
import numpy as np
import os
import pickle
from matplotlib.backends.backend_pdf import PdfPages

from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy


from sandbox.tuomas.mddpg.policies.stochastic_policy import \
    DummyExplorationStrategy

from sandbox.tuomas.mddpg.critics.gaussian_critic import MixtureGaussian2DCritic

#from sandbox.tuomas.mddpg.algos.vddpg import VDDPG
from rllab.misc.overrides import overrides
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
#plt.switch_backend('Agg')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('actor_lr', 0.01, 'Base learning rate for actor.')
flags.DEFINE_float('critic_lr', 0.001, 'Base learning rate for critic.')
flags.DEFINE_integer('path_length', 500, 'Maximum path length.')
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

    def __init__(self, *args, **kwargs):
        super(AlgTest, self).__init__(*args, **kwargs)

        self.lim_a = 2.
        self.lim_xy = 5
        self.plots_initialized = False
        self.eval_counter = 0
        self.n_training_paths = 10
        self.n_test_paths = 10
        self.training_path_counter = 0
        self.test_path_counter = 0

        self.training_com_list = []

        self.filename = lambda short_name, extension: (os.path.join(
            FLAGS.save_path,
            'exp001_' + short_name + '_iter' +
            str(self.eval_counter).zfill(6) + '.' + extension
        ))

        # Evaluate the Q function for the following states.
        # TODO: this list is not used.
        self.qpos_list = np.array([[0.0, 0.0]])

        self._init_plots()

    def eval_critic(self, o):
        xx = np.arange(-self.lim_a, self.lim_a, 0.05)
        X, Y = np.meshgrid(xx, xx)
        Z = np.zeros_like(X)
        all_actions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).transpose()
        obs = np.array([o] * all_actions.shape[0])

        feed = {
            self.qf.observations_placeholder: obs,
            self.qf.actions_placeholder: all_actions
        }
        Q = self.sess.run(self.qf.output, feed).reshape(X.shape)
        return X, Y, Q

    def rollout(self):
        env = self.env.wrapped_env
        com = []
        o = env.reset()
        for t in range(FLAGS.path_length):
            a, _ = self.policy.get_action(o)
            o, _, d, i = env.step(a)
            com.append(i['com'])
            if d:
                break
        return np.array(com)

    @overrides
    def plot_path(self, flush=False, **kwargs):
        if flush:
            com = np.array(self.training_com_list)
            line = self.h_training_paths[
                self.training_path_counter % self.n_training_paths]
            line.set_data(np.transpose(com[:, [0, 2]]))

            self.training_path_counter += 1
            self.training_com_list = []

        else:
            self.training_com_list.append(kwargs['info']['com'])


    def _init_plots(self):
        # Set up all critic plots.
        self._critic_fig = plt.figure(figsize=(7, 7))
        self.ax_critics = []
        n_plots = self.qpos_list.shape[0]
        for i in range(n_plots):
            ax = self._critic_fig.add_subplot(100 + n_plots * 10 + i + 1)
            self.ax_critics.append(ax)
            plt.axis('equal')
            ax.set_xlim((-self.lim_a, self.lim_a))
            ax.set_ylim((-self.lim_a, self.lim_a))

        # Setup actor plots
        self._actor_fig = plt.figure(figsize=(7, 7))
        self.ax_paths = self._actor_fig.add_subplot(111)
        #env.plot_env(self.ax_paths)
        plt.axis('equal')
        self.ax_paths.grid(True)
        self.ax_paths.set_xlim((-self.lim_xy, self.lim_xy))
        self.ax_paths.set_ylim((-self.lim_xy, self.lim_xy))

        self.h_training_paths = []
        self.h_test_paths = []
        for i in range(self.n_training_paths):
            self.h_training_paths += self.ax_paths.plot([], [], 'r')
        for i in range(self.n_test_paths):
            self.h_test_paths += self.ax_paths.plot([], [], 'b')

    def save_state(self):
        with open(self.filename('alg', 'pkl', 'wb')) as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    if __name__ == '__main__':
        @overrides
        def evaluate(self, epoch, es_path_returns):
            self._save_state()

            self.eval_counter += 1


            env = self.env
            while isinstance(env, ProxyEnv):
                env = env.wrapped_env

            #print(env.visitation_bins)

            for i in range(self.n_test_paths):
                line = self.h_test_paths[self.test_path_counter % self.n_test_paths]
                com = self.rollout()
                line.set_data(np.transpose(com[:, [0, 2]]))
                self.test_path_counter += 1


            plt.draw()
            plt.pause(0.001)

            for ax_critic, qpos in zip(self.ax_critics, self.qpos_list):
                env.reset_mujoco()
                obs = env.get_current_obs()

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

            # Save plots

            if FLAGS.save_path != '':
                self._critic_fig.savefig(self.filename('q', 'png'))
                self._actor_fig.savefig(self.filename('trajs', 'png'))



# -------------------------------------------------------------------
def test():
    from sandbox.tuomas.mddpg.kernels.gaussian_kernel import \
        SimpleAdaptiveDiagonalGaussianKernel
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.tuomas.mddpg.envs.hopper_env import HopperEnv

    #from sandbox.tuomas.mddpg.envs.multi_goal_env import MultiGoalEnv
    #from sandbox.tuomas.mddpg.envs.multi_goal_env import MultiGoalEnv
    from sandbox.tuomas.mddpg.critics.nn_qfunction import FeedForwardCritic


    alg_kwargs = dict(
        epoch_length=1 * FLAGS.path_length,  # evaluate / plot per SVGD step
        min_pool_size=1 * FLAGS.path_length,  # must be at least 2
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
            sample_dim=3,
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
            alpha=30,#100.,
        ))

    # ----------------------------------------------------------------------
    env = TfEnv(
        normalize(
            HopperEnv(
                random_init_state=False,
            ),
            clip=False
        )
    )

    es = DummyExplorationStrategy()
    #es = OUStrategy(env_spec=env.spec, mu=0, theta=0.15, sigma=0.03, clip=False)
    #es = OUStrategy(env_spec=env.spec, mu=0, theta=0.15, sigma=0.3, clip=False)

    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        observation_hidden_sizes=(),
        embedded_hidden_sizes=(100, 100),
    )

    #q_prior = MixtureGaussian2DCritic(
    #    "critic",
    #    observation_dim=env.observation_space.flat_dim,
    #    action_input=None,
    #    observation_input=None,
    #    weights=[1.],
    #    mus=[np.array((0., 0.))],
    #    sigmas=[1.0]
    #)

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
            q_prior=None, #q_prior,
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
