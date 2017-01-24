from rllab.envs.base import Env
from rllab import spaces
import tensorflow as tf
import numpy as np

from sandbox.tuomas.mddpg.policies.stochastic_policy import \
    DummyExplorationStrategy

from sandbox.tuomas.mddpg.critics.gaussian_critic import MixtureGaussian2DCritic
from sandbox.tuomas.mddpg.critics.nn_qfunction import MultiCritic

from sandbox.tuomas.mddpg.algos.vddpg import VDDPG
from rllab.misc.overrides import overrides
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate.')
flags.DEFINE_integer('modes', 2, 'Number of modes.')
flags.DEFINE_boolean('fixed', False, 'Fixed target distribution.')

K = 32  # number of particles
temperatures = np.array([[10.],
                         [1.]])

class MultimodalGaussianEnv(Env):

    def __init__(self, n_modes, action_lb, action_ub, fixed_observations):
        self._n_modes = n_modes
        self._action_lb = action_lb
        self._action_ub = action_ub
        self._fixed_observations = fixed_observations

    def step(self, action):
        return self.observe(), 0, True, {}

    def observe(self):
        temps = np.array([0.1, 1])
        temp = temps[[np.random.randint(0, 2)]]
        return temp

    def reset(self):
        return self.observe()

    @property
    def action_space(self):
        return spaces.Box(-1, 1, (2,))

    @property
    def observation_space(self):
        return spaces.Box(-1, 1, (1,))


class VDDPGTest(VDDPG):

    def eval_critic(self, o):
        xx = np.arange(-4, 4, 0.05)
        X, Y = np.meshgrid(xx, xx)
        all_actions = np.vstack([X.ravel(), Y.ravel()]).transpose()
        obs = np.array([o] * all_actions.shape[0])

        feed = self.qf.get_feed_dict(obs, all_actions)
        Q = self.sess.run(self.qf.output, feed).reshape(X.shape)
        return X, Y, Q

    @overrides
    def evaluate(self, epoch, es_path_returns):
        obs_single = self.pool.random_batch(1)['observations'].reshape((-1))
        X, Y, Q = self.eval_critic(obs_single)

        plt.clf()
        plt.contour(X, Y, Q, 20)
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))

        # Sample and plot actions. Note that we can actually interpolate over
        # the temperature: the policy is not trained for T = 3.
        self.plot_actions(obs_single, temperatures[0, 0], '*b')
        self.plot_actions(obs_single, 3, '*g')
        self.plot_actions(obs_single, temperatures[1, 0], '*r')

    def plot_actions(self, obs, temperature, style):
        N = 100
        all_obs = np.array([obs] * N)
        temps = np.array([temperature] * N)[:, None]
        all_actions = self.policy.get_actions(all_obs, temps)[0]

        for k, action in enumerate(all_actions):
            x = action[0]
            y = action[1]
            plt.plot(x, y, style)
            #ax_qf.text(x, y, '%d'%(k))

        plt.draw()
        plt.pause(0.001)


# -------------------------------------------------------------------
def test():
    from sandbox.haoran.mddpg.kernels.gaussian_kernel import \
        SimpleAdaptiveDiagonalGaussianKernel, \
        SimpleDiagonalConstructor, DiagonalGaussianKernel
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.tuomas.mddpg.policies.stochastic_policy \
        import StochasticNNPolicy


    adaptive_kernel = True

    ddpg_kwargs = dict(
        epoch_length=50,  # evaluate / plot per SVGD step
        min_pool_size=2,  # must be at least 2
        replay_pool_size=2,  # should only sample from recent experiences,
                        # note that the sample drawn can be one iteration older
        eval_samples=1,  # doesn't matter since we override evaluate()
        n_epochs=100000,  # number of SVGD steps
        policy_learning_rate=FLAGS.learning_rate,  # note: this is higher than DDPG's 1e-4
        qf_learning_rate=0.1,
        batch_size=32,  # only need recent samples, though it's slow
        alpha=1,  # 1 is the SVGD default
        train_actor=True,
        train_critic=False,
    )
    q_target_type = "none"  # do not update the critic

    # ----------------------------------------------------------------------

    env = TfEnv(MultimodalGaussianEnv(
        n_modes=FLAGS.modes,
        action_lb=np.array([-10, 10]),  # bounds on the particles
        action_ub=np.array([10, 10]),
        fixed_observations=True
    ))

    es = DummyExplorationStrategy()
    qf = MixtureGaussian2DCritic(
        "critic",
        observation_dim=env.observation_space.flat_dim,
        action_input=None,
        observation_input=None,
        weights=[1./3., 2./3.],
        mus=[np.array((-2., 0.)), np.array((1., 1.))],
        sigmas=[0.3, 1.0]
    )
    qf_temp = MultiCritic(
        critics=[qf],
    )

    policy = StochasticNNPolicy(
        'actor',
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        hidden_dims=(128, 128),
        sample_dim=2,
        temperature_dim=temperatures.shape[1],
        default_temperature=temperatures[0],
        freeze_samples=False
    )

    if adaptive_kernel:
        kernel = SimpleAdaptiveDiagonalGaussianKernel(
            "kernel",
            dim=env.action_space.flat_dim,
            h_min=0,
        )
    else:
        diag_constructor = SimpleDiagonalConstructor(
            dim=env.action_space.flat_dim,
            sigma=0.1,
        )
        kernel = DiagonalGaussianKernel(
            "kernel",
            diag=diag_constructor.diag(),
        )
    algorithm = VDDPGTest(
        env=env,
        exploration_strategy=es,
        policy=policy,
        kernel=kernel,
        qf=qf_temp,
        K=K,
        q_target_type=q_target_type,
        temperatures=temperatures,
        **ddpg_kwargs
    )
    algorithm.train()

if __name__ == "__main__":
    test() # this way variables defined in test() are not global, hence
    # not accessible to other methods, avoiding bugs
