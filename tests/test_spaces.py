from __future__ import print_function
from rllab.spaces import Product, Discrete
import numpy as np


def test_product_space():
    _ = Product([Discrete(3), Discrete(2)])
    product_space = Product(Discrete(3), Discrete(2))
    sample = product_space.sample()
    assert product_space.contains(sample)


def test_product_space_unflatten_n():
    space = Product([Discrete(3), Discrete(3)])
    np.testing.assert_array_equal(space.flatten((2,2)), space.flatten_n([(2, 2)])[0])
