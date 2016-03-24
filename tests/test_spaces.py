from __future__ import print_function
from nose2.tools import assert_true, assert_equal
import numpy as np


def test_product_space():
    from rllab.spaces import Product, Discrete
    _ = Product([Discrete(3), Discrete(2)])
    product_space = Product(Discrete(3), Discrete(2))
    sample = product_space.sample()
    print(sample)
    assert_true(product_space.contains(sample))
    # assert_equal(len(product_space.flatten(sample)), 2)
    # np.testing.assert_equal(product_space.unflatten(product_space.flatten(sample)), sample)
