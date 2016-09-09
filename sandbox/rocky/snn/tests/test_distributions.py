


from nose2.tools import such
from sandbox.rocky.snn.distributions.standard_bernoulli import StandardBernoulli
from sandbox.rocky.snn.distributions.bernoulli import Bernoulli
from sandbox.rocky.snn.distributions.standard_gaussian import StandardGaussian
import numpy as np
import theano
import theano.tensor as TT

with such.A("Standard Bernoulli") as it:
    @it.should("work")
    def test_standard_bernoulli():
        standard_bernoulli = StandardBernoulli(2)
        it.assertEqual(
            np.sum(standard_bernoulli.entropy(dist_info=dict(shape_placeholder=np.zeros((3, 2))))),
            np.log(2) * 6
        )

        placeholder = np.zeros((3, 2))
        old_info = new_info = dict(shape_placeholder=placeholder)
        x_val = np.zeros((3, 2))
        x_sym = theano.shared(x_val)
        placeholder_sym = theano.shared(placeholder)
        old_info_sym = new_info_sym = dict(shape_placeholder=placeholder_sym)
        np.testing.assert_array_equal(
            standard_bernoulli.kl_sym(old_info_sym, new_info_sym).eval(),
            np.zeros((3,))
        )
        np.testing.assert_array_equal(
            standard_bernoulli.kl(old_info, new_info),
            np.zeros((3,))
        )
        np.testing.assert_array_equal(
            standard_bernoulli.likelihood_ratio_sym(x_sym, old_info_sym, new_info_sym).eval(),
            np.ones((3,))
        )
        np.testing.assert_array_equal(
            standard_bernoulli.log_likelihood_sym(x_sym, old_info_sym).eval(),
            np.log(0.5) * 2 * np.ones((3,))
        )
        np.testing.assert_array_equal(
            standard_bernoulli.log_likelihood(x_val, old_info),
            np.log(0.5) * 2 * np.ones((3,))
        )

it.createTests(globals())

with such.A("Standard Gaussian") as it:
    @it.should("work")
    def test_standard_gaussian():
        standard_gaussian = StandardGaussian(2)
        it.assertEqual(
            np.sum(standard_gaussian.entropy(dist_info=dict(shape_placeholder=np.zeros((3, 2))))),
            0.5 * (np.log(2 * np.pi) + 1) * 6
        )

        placeholder = np.zeros((3, 2))
        old_info = new_info = dict(shape_placeholder=placeholder)
        x_val = np.zeros((3, 2))
        x_sym = theano.shared(x_val)
        placeholder_sym = theano.shared(placeholder)
        old_info_sym = new_info_sym = dict(shape_placeholder=placeholder_sym)
        np.testing.assert_array_equal(
            standard_gaussian.kl_sym(old_info_sym, new_info_sym).eval(),
            np.zeros((3,))
        )
        np.testing.assert_array_equal(
            standard_gaussian.kl(old_info, new_info),
            np.zeros((3,))
        )
        np.testing.assert_array_equal(
            standard_gaussian.likelihood_ratio_sym(x_sym, old_info_sym, new_info_sym).eval(),
            np.ones((3,))
        )
        np.testing.assert_array_equal(
            standard_gaussian.log_likelihood_sym(x_sym, old_info_sym).eval(),
            - np.log(2 * np.pi) * np.ones((3,))
        )
        np.testing.assert_array_equal(
            standard_gaussian.log_likelihood(x_val, old_info),
            - np.log(2 * np.pi) * np.ones((3,))
        )

it.createTests(globals())

with such.A("Bernoulli") as it:
    @it.should("work")
    def test_bernoulli():
        bernoulli = Bernoulli(3)

        new_p = np.array([[0.5, 0.5, 0.5], [.9, .9, .9]])
        old_p = np.array([[.9, .9, .9], [.1, .1, .1]])

        x = np.array([[1, 0, 1], [1, 1, 1]])

        x_sym = theano.shared(x)
        new_p_sym = theano.shared(new_p)
        old_p_sym = theano.shared(old_p)

        new_info = dict(p=new_p)
        old_info = dict(p=old_p)

        new_info_sym = dict(p=new_p_sym)
        old_info_sym = dict(p=old_p_sym)

        np.testing.assert_allclose(
            np.sum(bernoulli.entropy(dist_info=new_info)),
            np.sum(- new_p * np.log(new_p + 1e-8) - (1 - new_p) * np.log(1 - new_p + 1e-8)),
        )

        np.testing.assert_allclose(
            np.sum(bernoulli.kl_sym(old_info_sym, new_info_sym).eval()),
            np.sum(old_p * (np.log(old_p + 1e-8) - np.log(new_p + 1e-8)) + (1 - old_p) * (np.log(1 - old_p + 1e-8) -
                                                                                          np.log(1 - new_p + 1e-8))),
        )
        np.testing.assert_allclose(
            np.sum(bernoulli.kl(old_info, new_info)),
            np.sum(old_p * (np.log(old_p + 1e-8) - np.log(new_p + 1e-8)) + (1 - old_p) * (np.log(1 - old_p + 1e-8) -
                                                                                          np.log(1 - new_p + 1e-8))),
        )
        np.testing.assert_allclose(
            bernoulli.likelihood_ratio_sym(x_sym, old_info_sym, new_info_sym).eval(),
            np.prod((x * new_p + (1 - x) * (1 - new_p)) / (x * old_p + (1 - x) * (1 - old_p) + 1e-8), axis=-1)
        )
        np.testing.assert_allclose(
            bernoulli.log_likelihood_sym(x_sym, old_info_sym).eval(),
            np.sum(x * np.log(old_p + 1e-8) + (1 - x) * np.log(1 - old_p + 1e-8), axis=-1)
        )
        np.testing.assert_allclose(
            bernoulli.log_likelihood(x, old_info),
            np.sum(x * np.log(old_p + 1e-8) + (1 - x) * np.log(1 - old_p + 1e-8), axis=-1)
        )

it.createTests(globals())
