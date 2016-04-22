from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.snn.core.network import StochasticMLP
from sandbox.rocky.snn.core.lasagne_layers import IndependentGaussianLayer, IndependentBernoulliLayer, GaussianLayer, \
    BernoulliLayer
from nose2.tools import such

with such.A("Stochastic MLP") as it:
    @it.should("construct latent input properly")
    def test_construct_latent_input():
        mlp = StochasticMLP(
            input_shape=(5,),
            input_latent_vars=(('independent_gaussian', 5), ('independent_bernoulli', 3)),
            hidden_sizes=tuple(),
            output_dim=2,
        )
        it.assertEqual(len(mlp.latent_layers), 2)
        it.assertIsInstance(mlp.latent_layers[0], IndependentGaussianLayer)
        it.assertIsInstance(mlp.latent_layers[1], IndependentBernoulliLayer)
        it.assertEqual(mlp.latent_layers[0].num_units, 5)
        it.assertEqual(mlp.latent_layers[1].num_units, 3)
        it.assertEqual(mlp.latent_dims, [5, 3])


    @it.should("construct latent hidden vars properly")
    def test_construct_latent_hidden():
        mlp = StochasticMLP(
            input_shape=(5,),
            hidden_latent_vars=[[('gaussian', 5), ('bernoulli', 3)], []],
            hidden_sizes=(32, 32),
            output_dim=2,
        )
        it.assertEqual(len(mlp.latent_layers), 2)
        it.assertIsInstance(mlp.latent_layers[0], GaussianLayer)
        it.assertIsInstance(mlp.latent_layers[1], BernoulliLayer)
        it.assertEqual(mlp.latent_layers[0].num_units, 5)
        it.assertEqual(mlp.latent_layers[1].num_units, 3)
        it.assertEqual(mlp.latent_dims, [5, 3])

it.createTests(globals())
