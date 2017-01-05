"""
Test whether our SVGD code in MDDPG is correct.

The current code is slow.

Setting:
- env: horizon = 1, single observation (default a single number 1),
    action is a 1D point
- policy: action is a function of the observation (NN); if the NN is linear
    then we are just doing SVGD; otherwise we are multiplying the jacobian of
    NN output w.r.t. parameters with the SVGD step
- critic: the Q value is fixed and equals log p, where p is a
    mixture of Gaussians
- algo: evaluate() is overwritten to plot p and q
- kernel: diagonal Guassian; the variance can be fixed or adaptive
"""
from rllab.envs.base import Env
from rllab import spaces
import tensorflow as tf
import numpy as np

class OneStepEnv(Env):
    def __init__(self, observation_dim,
        action_lb, action_ub, fixed_observation):
        self.observation_dim = observation_dim
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.fixed_observation = fixed_observation

    def step(self, action):
        return self.fixed_observation, 0, True, {}

    def reset(self):
        return self.fixed_observation

    @property
    def action_space(self):
        return spaces.Box(self.action_lb, self.action_ub)

    @property
    def observation_space(self):
        return spaces.Box(self.fixed_observation, self.fixed_observation)

    def render(self):
        raise NotImplementedError

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

from sandbox.haoran.mddpg.qfunctions.nn_qfunction import NNCritic
from rllab.core.serializable import Serializable
class MixtureGaussianCritic(NNCritic):
    """ Q(s,a) is a 1D mixture of Gaussian in a """
    def __init__(
            self,
            scope_name,
            observation_dim,
            action_input,
            weights, mus, sigmas,
            reuse=False,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.weights = weights
        self.mus = mus
        self.sigmas = sigmas

        super(MixtureGaussianCritic, self).__init__(
            scope_name=scope_name,
            observation_dim=observation_dim,
            action_dim=1,
            action_input=action_input,
            reuse=reuse,
            **kwargs
        )

    def create_network(self, action_input):
        # unnormalized density
        output = tf.log(tf.add_n([
            w * (1./tf.sqrt(2. * np.pi * tf.square(sigma)) *
                tf.exp(-0.5/tf.square(sigma) * tf.square(action_input-mu)))
            for w, mu, sigma in zip(self.weights, self.mus, self.sigmas)
        ]))
        return output

    def get_weight_tied_copy(self, action_input):
        """
        HT: basically, re-run __init__ with specified kwargs. In particular,
        the variable scope doesn't change, and self.observations_placeholder
        and NN params are reused.
        """
        return self.__class__(
            scope_name=self.scope_name,
            observation_dim=self.observation_dim,
            action_input=action_input,
            reuse=True,
            weights=self.weights,
            mus=self.mus,
            sigmas=self.sigmas,
        )

from sandbox.haoran.mddpg.policies.mnn_policy import \
    FeedForwardMultiPolicy, MNNStrategy
class FeedForwardMultiPolicyTest(FeedForwardMultiPolicy):
    def __init__(
        self,
        scope_name,
        observation_dim,
        action_dim,
        K,
        shared_hidden_sizes=(8,),
        independent_hidden_sizes=tuple(),
        hidden_W_init=None,
        hidden_b_init=None,
        output_W_init=None,
        output_b_init=None,
        hidden_nonlinearity=tf.identity,
        output_nonlinearity=tf.identity,
        scalar_network=True,
    ):
        """
        Essentially the same as FeedForwardMultiPolicy, except that turning on
        scalar_network makes each head essentially a single scalar and we only
        need to learn that scalar. This should produce identical results to the
        usual SVGD.

        Turning off scalar_network allows gradients to be propagated to network
        parameters. This way we can test whether the jacobian is computed and
        used correctly. Since NNs have high representation power, the ultimate
        distribution should be identical to scalar networks. However, different
        network structures and learning rates may lead to different local
        optima.
        """
        Serializable.quick_init(self, locals())
        if scalar_network:
            hidden_W_init = tf.constant_initializer(0.)
            hidden_b_init = tf.constant_initializer(0.)
            output_W_init = tf.constant_initializer(0.)
            output_b_init = tf.random_uniform_initializer(-3, -2)
            # modify output_b_init to change the initial particle distribution
        super(FeedForwardMultiPolicyTest, self).__init__(
            scope_name,
            observation_dim,
            action_dim,
            K,
            shared_hidden_sizes,
            independent_hidden_sizes,
            hidden_W_init,
            hidden_b_init,
            output_W_init,
            output_b_init,
            hidden_nonlinearity,
            output_nonlinearity,
        )
class OneSampleReplayPool(object):
    def __init__(self):
        pass

    def add_sample(self, observation, action, reward, terminal,
                   final_state):
        self._observations = np.array([observation])
        self._actions = np.array([action])
        self._rewards = np.array([reward])
        self._terminals = np.array([terminal])
        self._final_state = np.array([final_state])

    def random_batch(self, batch_size):
        return dict(
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            terminals=self._terminals,
            next_observations=self._observations
        )
    @property
    def size(self):
        return 2 # lie

from sandbox.haoran.mddpg.algos.mddpg import MDDPG
from rllab.misc.overrides import overrides
import matplotlib.pyplot as plt
class MDDPGTest(MDDPG):
    @overrides
    def evaluate(self, epoch, es_path_returns):
        obs = np.array([self.env.reset()])
        feed_dict = {
            self.policy.observations_placeholder: obs
        }
        # xs is the list of particles (scalars)
        xs = self.sess.run(self.policy.output, feed_dict).ravel()
        plt.clf()
        xx = np.linspace(-5, 5, num=100)
        yy = np.zeros_like(xx)
        # fixed density kernel size
        # q_sigma = 0.1
        # adaptive density kernel size
        q_sigma = 1./np.sqrt(self.kernel.diags[0])
        for p in xs:
            yy += (np.exp(-0.5/(q_sigma**2) * (xx-p)**2) *
                1./np.sqrt(2. * np.pi * q_sigma ** 2))
        yy /= len(xs)
        plt.plot(xx,yy,'r-.')

        ws = self.qf.weights
        mus = self.qf.mus
        sigmas = self.qf.sigmas
        yy_target = np.zeros_like(xx)
        for w, mu, sigma in zip(self.qf.weights, self.qf.mus, self.qf.sigmas):
            yy_target += (w * 1./np.sqrt(2. * np.pi * sigma**2) *
                np.exp(-0.5/ (sigma**2) * (xx - mu) ** 2))
        plt.plot(xx, yy_target, 'b-')

        plt.draw()
        plt.legend(['q','p'])
        plt.xlim([-3, 3])
        plt.ylim([0,0.5])
        plt.pause(0.001)


# -------------------------------------------------------------------
def test():
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from sandbox.haoran.mddpg.kernels.gaussian_kernel import \
        SimpleAdaptiveDiagonalGaussianKernel, \
        SimpleDiagonalConstructor, DiagonalGaussianKernel
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler

    K = 20 # number of particles

    adaptive_kernel = True

    # three modes
    # weights = np.array([1./4., 1./2., 1./4.],np.float32)
    # mus = np.array([-2., 0., 2.],np.float32)
    # sigmas = np.array([0.3, 0.3, 0.3],np.float32)

    # two modes
    weights = np.array([1./3., 2./3.],np.float32)
    mus = np.array([-2., 2.],np.float32)
    sigmas = np.array([1., 1.],np.float32)

    # one mode
    # weights = np.array([1.],np.float32)
    # mus = np.array([0.],np.float32)
    # sigmas = np.array([1.],np.float32)

    ddpg_kwargs = dict(
        epoch_length = 1, # evaluate / plot per SVGD step
        min_pool_size = 2, # must be at least 2
        replay_pool_size=2, # should only sample from recent experiences,
                        # note that the sample drawn can be one iteration older
        eval_samples = 1, # doesn't matter since we override evaluate()
        n_epochs=1000, # number of SVGD steps
        policy_learning_rate=0.1, # note: this is higher than DDPG's 1e-4
        batch_size=1, # only need recent samples, though it's slow
        alpha=1, # 1 is the SVGD default
    )
    q_target_type = "none" # do not update the critic

    # ----------------------------------------------------------------------
    env = TfEnv(OneStepEnv(
        observation_dim=1,
        action_lb=np.array([-10]), # bounds on the particles
        action_ub=np.array([10]),
        fixed_observation=np.array([1]),
    ))

    es = MNNStrategy(
        K=K,
        substrategy=OUStrategy(env_spec=env.spec),
        switch_type="per_path"
    )
    qf = MixtureGaussianCritic(
        "critic",
        observation_dim=env.observation_space.flat_dim,
        action_input=None,
        weights=weights,
        mus=mus,
        sigmas=sigmas,
    )

    policy = FeedForwardMultiPolicyTest(
        "actor",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        K=K,
        shared_hidden_sizes=tuple(),
        independent_hidden_sizes=tuple(),
        scalar_network=True,
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
    algorithm = MDDPGTest(
        env=env,
        exploration_strategy=es,
        policy=policy,
        kernel=kernel,
        qf=qf,
        K=K,
        q_target_type=q_target_type,
        **ddpg_kwargs
    )
    # algorithm.pool = OneSampleReplayPool()
    algorithm.train()

if __name__ == "__main__":
    test() # this way variables defined in test() are not global, hence
    # not accessible to other methods, avoiding bugs
