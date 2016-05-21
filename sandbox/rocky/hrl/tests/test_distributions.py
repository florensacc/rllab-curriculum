from __future__ import print_function
from __future__ import absolute_import

from nose2.tools import such
from sandbox.rocky.hrl.distributions.product_distribution import productdistribution
from rllab.distributions.categorical import categorical
from rllab.distributions.diagonal_gaussian import diagonalgaussian
import theano
import theano.tensor as tt
from rllab.misc import special
import numpy as np


def random_softmax(ndim):
    x = np.random.uniform(size=(ndim,))
    x = x - np.max(x)
    x = x / np.sum(np.exp(x))
    return x


with such.a("product distribution") as it:
    dist1 = productdistribution([categorical(), categorical()], [5, 3])
    dist2 = productdistribution([diagonalgaussian(), dist1], [5, 8])


    @it.should
    def test_dist_info_keys():
        it.assertequal(set(dist1.dist_info_keys), {"id_0_prob", "id_1_prob"})
        it.assertequal(set(dist2.dist_info_keys), {"id_0_mean", "id_0_log_std",
                                                   "id_1_id_0_prob", "id_1_id_1_prob"})


    @it.should
    def test_kl_sym():
        old_id_0_prob = random_softmax(5)
        old_id_1_prob = random_softmax(3)
        new_id_0_prob = random_softmax(5)
        new_id_1_prob = random_softmax(3)
        old_dist_info_vars = dict(
            id_0_prob=theano.shared(old_id_0_prob),
            id_1_prob=theano.shared(old_id_1_prob)
        )
        new_dist_info_vars = dict(
            id_0_prob=theano.shared(new_id_0_prob),
            id_1_prob=theano.shared(new_id_1_prob)
        )
        np.testing.assert_allclose(
            dist1.kl_sym(old_dist_info_vars, new_dist_info_vars).eval(),
            categorical().kl_sym(dict(prob=old_id_0_prob), dict(prob=new_id_0_prob)).eval() +
            categorical().kl_sym(dict(prob=old_id_1_prob), dict(prob=new_id_1_prob)).eval()
        )

it.createtests(globals())
