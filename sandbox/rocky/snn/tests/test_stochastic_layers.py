from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.snn.core.lasagne_layers import IndependentGaussianLayer, IndependentBernoulliLayer, GaussianLayer, \
    BernoulliLayer

import lasagne.layers as L
from rllab.core.lasagne_helpers import get_full_output
import numpy as np
from rllab.misc import ext
from nose2.tools import such

with such.A("Stochastic layers") as it:
    @it.should("work")
    def test_construct():
        l_in = L.InputLayer(shape=(None, 5))
        l_gaussian = IndependentGaussianLayer(l_in, num_units=3)
        l_bernoulli = IndependentBernoulliLayer(l_in, num_units=4)
        l_cond_gaussian = GaussianLayer(l_in, num_units=5)
        l_cond_bernoulli = BernoulliLayer(l_in, num_units=6)

        outs = L.get_output([l_gaussian, l_bernoulli, l_cond_gaussian, l_cond_bernoulli])

        f_outs = ext.compile_function([l_in.input_var], outs)

        out_val = f_outs(np.zeros((2, 5)))
        it.assertEqual(out_val[0].shape, (2, 3))
        it.assertEqual(out_val[1].shape, (2, 4))
        it.assertEqual(out_val[2].shape, (2, 5))
        it.assertEqual(out_val[3].shape, (2, 6))


    @it.should("output logli information")
    def test_logli():
        l_in = L.InputLayer(shape=(None, 5))
        l_cond_bernoulli = IndependentBernoulliLayer(l_in, num_units=6)
        cond_berboulli, extras = get_full_output(l_cond_bernoulli)
        dist = extras[l_cond_bernoulli]["distribution"]
        f_out = ext.compile_function(
            [l_in.input_var],
            dist.log_likelihood_sym(cond_berboulli, extras[l_cond_bernoulli]["dist_info"])
        )
        out_val = f_out(np.zeros((2, 5)))
        np.testing.assert_allclose(np.mean(out_val), np.log(0.5) * 6)

it.createTests(globals())
