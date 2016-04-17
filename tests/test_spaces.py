from __future__ import print_function


def test_product_space():
    from rllab.spaces import Product, Discrete
    _ = Product([Discrete(3), Discrete(2)])
    product_space = Product(Discrete(3), Discrete(2))
    sample = product_space.sample()
    assert product_space.contains(sample)
