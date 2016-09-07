


from nose2.tools import such
from rllab.envs.base import EnvSpec
from rllab.spaces.box import Box
from sandbox.rocky.snn.policies.stochastic_gaussian_mlp_policy import StochasticGaussianMLPPolicy
import numpy as np

with such.A("Stochastic MLP") as it:
    @it.should("work")
    def test_stochastic_policy():
        env_spec = EnvSpec(
            observation_space=Box(low=-1, high=1, shape=(5,)),
            action_space=Box(low=-1, high=1, shape=(3,))
        )
        policy = StochasticGaussianMLPPolicy(
            env_spec=env_spec,
            input_latent_vars=[('independent_bernoulli', 3)],
            hidden_sizes=(32, 32),
            hidden_latent_vars=[
                [('bernoulli', 3), ('gaussian', 3, dict(reparameterize=True))],
                [('bernoulli', 3), ('gaussian', 3, dict(reparameterize=False))],
            ],
        )
        it.assertEqual(policy._n_latent_layers, 5)

        action, agent_info = policy.get_action(np.zeros(5, ))
        it.assertEqual(action.shape, (3,))
        it.assertSetEqual(
            {
                "mean", "log_std", "latent_0_shape_placeholder", "latent_1_p", "latent_2_epsilon",
                "latent_2_shape_placeholder",
                "latent_3_p", "latent_4_mean", "latent_4_log_std", "latent_0", "latent_1", "latent_2", "latent_3",
                "latent_4"
            },
            set(agent_info.keys())
        )

it.createTests(globals())
