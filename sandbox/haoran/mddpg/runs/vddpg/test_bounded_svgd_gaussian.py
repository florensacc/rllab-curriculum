from rllab.envs.base import Env
from rllab import spaces
import tensorflow as tf
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Base learning rate.')
flags.DEFINE_integer('modes', 2, 'Number of modes.')
flags.DEFINE_boolean('fixed', True, 'Fixed target distribution.')
flags.DEFINE_string('svgd_target', 'action', 'Where SVGD operates.')


class MultimodalGaussianEnv(Env):

    def __init__(self, n_modes, action_lb, action_ub, fixed_observations):
        self._n_modes = n_modes
        self._action_lb = action_lb
        self._action_ub = action_ub
        self._fixed_observations = fixed_observations

    def step(self, action):
        return self.observe(), 0, True, {}

    def observe(self):
        if self._fixed_observations is not None:
            n_cases = self._fixed_observations.shape[0]
            i = np.random.randint(low=0, high=n_cases)
            return self._fixed_observations[i]

        modes = self._n_modes

        weights = np.random.rand(modes)
        weights = weights / weights.sum()
        mus = np.random.randn(modes)*2 - 1
        sigmas = np.random.rand(modes) + 0.5

        obs = np.stack((weights, mus, sigmas), axis=1) # one row per mode
        obs = np.ndarray.flatten(obs) # (w_1, mu_1, sigma_1, w_2, ...)

        return obs

    def reset(self):
        return self.observe()

    @property
    def horizon(self):
        pass

    @property
    def action_space(self):
        return spaces.Box(self._action_lb, self._action_ub)

    @property
    def observation_space(self):
        dummy = np.array([0] * self._n_modes * 3)
        return spaces.Box(dummy, dummy)


from sandbox.haoran.mddpg.qfunctions.nn_qfunction import NNCritic
from rllab.core.serializable import Serializable
class MixtureGaussianCritic(NNCritic):
    """ Q(s,a) is a 1D mixture of Gaussian in a """
    def __init__(
            self,
            scope_name,
            observation_dim,
            action_input,
            observation_input,
            reuse=False,
            **kwargs
    ):
        assert observation_dim % 3 == 0
        Serializable.quick_init(self, locals())
        super(MixtureGaussianCritic, self).__init__(
            scope_name=scope_name,
            observation_dim=observation_dim,
            action_dim=1,
            action_input=action_input,
            observation_input=observation_input,
            reuse=reuse,
            **kwargs
        )

    def create_network(self, action_input, observation_input):
        obs = tf.unstack(observation_input, axis=1)
        weights = obs[0::3] # a list of [w_i^1, w_i^2,...], where
        # i is the index of the mode, the superscript indexes the sample.
        # So we assume that all observations have the same n_modes
        mus = obs[1::3]
        sigmas = obs[2::3]

        # TODO: assume all samples in a batch are the same (use index 0).
        # unnormalized density
        output = tf.log(tf.add_n([
            w[0] * (1./tf.sqrt(2. * np.pi * tf.square(sigma[0])) *
                tf.exp(-0.5/tf.square(sigma[0])*tf.square(action_input-mu[0])))
            for w, mu, sigma in zip(weights, mus, sigmas)
        ]))
        return output


    def get_weight_tied_copy(self, action_input, observation_input):
        """
        HT: basically, re-run __init__ with specified kwargs. In particular,
        the variable scope doesn't change, and self.observations_placeholder
        and NN params are reused.
        """
        return self.__class__(
            scope_name=self.scope_name,
            observation_dim=self.observation_dim,
            action_input=action_input,
            observation_input=observation_input,
            reuse=True,
        )

from sandbox.tuomas.mddpg.policies.stochastic_policy import \
    DummyExplorationStrategy

from sandbox.tuomas.mddpg.algos.vddpg import VDDPG
from rllab.misc.overrides import overrides
import matplotlib.pyplot as plt
class VDDPGTest(VDDPG):
    @overrides
    def evaluate(self, epoch, es_path_returns):
        obs_single = self.pool.random_batch(1)['observations']

        obs = np.tile(obs_single, (100, 1))
        feed_dict = self.policy.get_feed_dict(obs)

        xs = self.sess.run(self.policy.output, feed_dict).ravel()
        plt.clf()
        xx = np.linspace(-1, 1, num=100)
        yy = np.zeros_like(xx)
        # fixed density kernel size
        q_sigma = 0.1
        # adaptive density kernel size
        # q_sigma = 1./np.sqrt(self.kernel.diags[0])
        for p in xs:
            yy += (np.exp(-0.5/(q_sigma**2) * (xx-p)**2) *
                1./np.sqrt(2. * np.pi * q_sigma ** 2))
        delta = xx[1] - xx[0]
        yy = yy / (np.sum(yy) * delta)
        plt.plot(xx,yy,'r-.')

        ws = obs_single[0,0::3]
        mus = obs_single[0,1::3]
        sigmas = obs_single[0,2::3]

        yy_target = np.zeros_like(xx)
        #for w, mu, sigma in zip(self.qf.weights, self.qf.mus, self.qf.sigmas):
        for w, mu, sigma in zip(ws, mus, sigmas):
            yy_target += (w * 1./np.sqrt(2. * np.pi * sigma**2) *
                np.exp(-0.5/ (sigma**2) * (xx - mu) ** 2))
        yy_target = yy_target / (np.sum(yy_target) * delta)
        plt.plot(xx, yy_target, 'b-')

        plt.plot(xs, np.zeros_like(xs), 'r*')
        plt.draw()
        plt.legend(['q','p','x'])
        plt.xlim([-1, 1])
        plt.ylim([0, 2. * np.max(yy_target)])
        plt.pause(0.001)


# -------------------------------------------------------------------
def test():
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from sandbox.haoran.mddpg.kernels.gaussian_kernel import \
        SimpleAdaptiveDiagonalGaussianKernel, \
        SimpleDiagonalConstructor, DiagonalGaussianKernel
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
    from sandbox.tuomas.mddpg.policies.stochastic_policy \
        import StochasticNNPolicy

    K = 10 # number of particles

    adaptive_kernel = True

    ddpg_kwargs = dict(
        epoch_length=10,  # evaluate / plot per SVGD step
        min_pool_size=2,  # must be at least 2
        replay_pool_size=2,  # should only sample from recent experiences,
                        # note that the sample drawn can be one iteration older
        eval_samples=1,  # doesn't matter since we override evaluate()
        n_epochs=100000,  # number of SVGD steps
        policy_learning_rate=FLAGS.learning_rate,  # note: this is higher than DDPG's 1e-4
        batch_size=64,  # only need recent samples, though it's slow
        alpha=1,  # 1 is the SVGD default; don't change it
        train_critic=False,
        svgd_target=FLAGS.svgd_target,
    )
    q_target_type = "none"  # do not update the critic

    # ----------------------------------------------------------------------
    if FLAGS.fixed:
        if FLAGS.modes == 1:
            # near uniform
            fixed_observations = np.array([
                [1., 0., 1.],
            ])
        elif FLAGS.modes == 2:
            # two modes, two cases
            fixed_observations = np.array([
                [1./3., -2., 1.,    2./3., 2., 1.],
                # [2./3., -2., 1.,    1./3., 2., 1.],
            ])
        elif FLAGS.modes == 3:
            # three modes, three cases
            fixed_observations = np.array([
                [1./3., -4., 1.,    1./3., 0., 1.,    1./3., 4., 1.],
                [1./2., -4., 1.,    1./4., 0., 1.,    1./4., 4., 1.],
                [1./4., -4., 1.,    1./4., 0., 1.,    1./2., 4., 1.],
            ])
        else:
            raise NotImplementedError
    else:
        fixed_observations = None

    env = TfEnv(MultimodalGaussianEnv(
        n_modes=FLAGS.modes,
        action_lb=np.array([-1]), # bounds on the particles
        action_ub=np.array([1]),
        fixed_observations=fixed_observations
    ))

    #es = MNNStrategy(
    #    K=K,
    #    substrategy=OUStrategy(env_spec=env.spec),
    #    switch_type="per_path"
    #)

    es = DummyExplorationStrategy()
    qf = MixtureGaussianCritic(
        "critic",
        observation_dim=env.observation_space.flat_dim,
        action_input=None,
        observation_input=None,
    )

    policy = StochasticNNPolicy(
        'actor',
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        hidden_dims=(128, 128),
        sample_dim=1,
        freeze_samples=False,
        output_nonlinearity=tf.nn.tanh,
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
        qf=qf,
        q_prior=None,
        K=K,
        q_target_type=q_target_type,
        **ddpg_kwargs
    )
    algorithm.train()

if __name__ == "__main__":
    test() # this way variables defined in test() are not global, hence
    # not accessible to other methods, avoiding bugs
