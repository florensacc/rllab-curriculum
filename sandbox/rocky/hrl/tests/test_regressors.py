from __future__ import print_function
from __future__ import absolute_import

from nose2.tools import such
from sandbox.rocky.hrl.regressors.shared_network_auto_mlp_regressor import SharedNetworkAutoMLPRegressor, \
    space_to_distribution, output_to_info
from rllab.distributions.categorical import Categorical
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
from rllab.spaces.product import Product
from sandbox.rocky.hrl.distributions.product_distribution import ProductDistribution
import numpy as np
import theano
import theano.tensor as TT

with such.A("Shared Network Auto MLP Regressor") as it:
    @it.should
    def test_space_to_distribution():
        dist1 = space_to_distribution(Discrete(5))
        it.assertIsInstance(dist1, Categorical)
        dist2 = space_to_distribution(Box(low=-1, high=1, shape=(5,)))
        it.assertIsInstance(dist2, DiagonalGaussian)
        dist3 = space_to_distribution(Product(Discrete(5), Box(low=-1, high=1, shape=(3,))))
        it.assertIsInstance(dist3, ProductDistribution)
        it.assertIsInstance(dist3.distributions[0], Categorical)
        it.assertIsInstance(dist3.distributions[1], DiagonalGaussian)
        it.assertEqual(dist3.dimensions, [5, 3])
        dist4 = space_to_distribution(Product(Discrete(5), Product(Discrete(5), Discrete(3))))
        it.assertIsInstance(dist4.distributions[1].distributions[0], Categorical)


    @it.should
    def test_output_to_info():
        sample = TT.ones((1, 10))
        np.testing.assert_array_equal(
            output_to_info(sample, Discrete(10))["prob"].eval(),
            np.ones((1, 10), dtype=theano.config.floatX) / 10
        )

        sample = TT.ones((1, 10))
        info_sym = output_to_info(sample, Product(Discrete(8), Discrete(2)))
        it.assertEqual(
            set(info_sym.keys()),
            {"id_0_prob", "id_1_prob"}
        )
        np.testing.assert_array_equal(
            info_sym["id_0_prob"].eval(),
            np.ones((1, 8), dtype=theano.config.floatX) / 8,
        )
        np.testing.assert_array_equal(
            info_sym["id_1_prob"].eval(),
            np.ones((1, 2), dtype=theano.config.floatX) / 2,
        )

it.createTests(globals())
