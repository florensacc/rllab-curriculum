import functools
import itertools
from collections import defaultdict
from contextlib import contextmanager

import tensorflow as tf
import numpy as np
import prettytensor as pt
from prettytensor.pretty_tensor_class import Layer
from progressbar import ProgressBar

from rllab.core.serializable import Serializable
from rllab.misc.ext import AttrDict, extract
from rllab.misc.overrides import overrides
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import CustomPhase, resconv_v1_customconv, plstmconv_v1, int_shape, \
    universal_int_shape, get_linear_ar_mask, tf_go, safe_log
import sandbox.pchen.InfoGAN.infogan.misc.imported.scopes as scopes
import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
import rllab.misc.logger as logger

TINY = 1e-8

floatX = np.float32

seeds = []

@contextmanager
def set_current_seed(seed):
    seeds.append(seed)
    yield
    seeds.pop(len(seeds) - 1)


def get_current_seed():
    return seeds[-1] if len(seeds) > 0 else None

INSPECT = defaultdict(list)

class Distribution(Serializable):
    @property
    def dist_flat_dim(self):
        """
        :rtype: int
        """
        raise NotImplementedError

    @property
    def dim(self):
        """
        :rtype: int
        """
        raise NotImplementedError

    @property
    def effective_dim(self):
        """
        The effective dimension when used for rescaling quantities. This can be different from the
        actual dimension when the actual values are using redundant representations (e.g. for categorical
        distributions we encode it in onehot representation)
        :rtype: int
        """
        raise NotImplementedError

    def kl_prior(self, dist_info):
        return self.kl(dist_info, self.prior_dist_info(list(dist_info.values())[0].get_shape()[0]))

    def logli(self, x_var, dist_info):
        """
        :param x_var:
        :param dist_info:
        :return: log likelihood of the data
        """
        raise NotImplementedError

    def logli_prior(self, x_var):
        return self.logli(
            x_var,
            self.prior_dist_info(x_var.get_shape()[0])
        )

    def logli_init_prior(self, x_var):
        return self.logli(x_var, self.prior_dist_info(x_var.get_shape()[0]))

    def nonreparam_logli(self, x_var, dist_info):
        """
        :param x_var:
        :param dist_info:
        :return: the non-reparameterizable part of the log likelihood
        """
        raise NotImplementedError

    def activate_dist(self, flat_dist):
        """
        :param flat_dist: flattened dist info without applying nonlinearity yet
        :return: a dictionary of dist infos
        """
        raise NotImplementedError

    def deactivate_dist(self, dict):
        """
        inverse of activate_dist
        """
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        """
        :rtype: list[str]
        """
        raise NotImplementedError

    def entropy(self, dist_info):
        """
        :return: entropy for each minibatch entry
        """
        raise NotImplementedError

    def marginal_entropy(self, dist_info):
        """
        :return: the entropy of the mixture distribution averaged over all minibatch entries. Will return in the same
        shape as calling `:code:Distribution.entropy`
        """
        raise NotImplementedError

    def marginal_logli(self, x_var, dist_info):
        """
        :return: the log likelihood of the given variable under the mixture distribution averaged over all minibatch
        entries.
        """
        raise NotImplementedError

    def sample(self, dist_info):
        return self.sample_logli(dist_info)[0]

    def sample_logli(self, dist_info):
        raise NotImplementedError

    def sample_prior(self, batch_size):
        return self.sample(self.prior_dist_info(batch_size))

    def sample_init_prior(self, batch_size):
        return self.sample(self.init_prior_dist_info(batch_size))

    def prior_dist_info(self, batch_size):
        """
        :return: a dictionary containing distribution information about the standard prior distribution, the shape
                 of which is jointly decided by batch_size and self.dim
        """
        raise NotImplementedError

    def init_mode(self):
        pass

    def train_mode(self):
        pass


class Categorical(Distribution):
    def __init__(self, dim):
        Serializable.quick_init(self, locals())
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self.dim

    @property
    def effective_dim(self):
        return 1

    def logli(self, x_var, dist_info):
        prob = dist_info["prob"]
        return tf.reduce_sum(tf.log(prob + TINY) * x_var, reduction_indices=1)

    def prior_dist_info(self, batch_size):
        prob = tf.ones([batch_size, self.dim]) * floatX(1.0 / self.dim)
        return dict(prob=prob)

    def marginal_logli(self, x_var, dist_info):
        prob = dist_info["prob"]
        avg_prob = tf.tile(
            tf.reduce_mean(prob, reduction_indices=0, keep_dims=True),
            tf.pack([tf.shape(prob)[0], 1])
        )
        return self.logli(x_var, dict(prob=avg_prob))

    def nonreparam_logli(self, x_var, dist_info):
        return self.logli(x_var, dist_info)

    def kl(self, p, q):
        """
        :param p: left dist info
        :param q: right dist info
        :return: KL(p||q)
        """
        p_prob = p["prob"]
        q_prob = q["prob"]
        return tf.reduce_sum(
            p_prob * (tf.log(p_prob + TINY) - tf.log(q_prob + TINY)),
            reduction_indices=1
        )

    def sample(self, dist_info):
        prob = dist_info["prob"]
        ids = tf.multinomial(tf.log(prob + TINY), num_samples=1, seed=get_current_seed())[:, 0]
        onehot = tf.constant(np.eye(self.dim, dtype=np.float32))
        return tf.nn.embedding_lookup(onehot, ids)

    def activate_dist(self, flat_dist):
        return dict(prob=tf.nn.softmax(flat_dist))

    def entropy(self, dist_info):
        prob = dist_info["prob"]
        return -tf.reduce_sum(prob * tf.log(prob + TINY), reduction_indices=1)

    def marginal_entropy(self, dist_info):
        prob = dist_info["prob"]
        avg_prob = tf.tile(
            tf.reduce_mean(prob, reduction_indices=0, keep_dims=True),
            tf.pack([tf.shape(prob)[0], 1])
        )
        return self.entropy(dict(prob=avg_prob))

    @property
    def dist_info_keys(self):
        return ["prob"]


G_IDX = 0


class Gaussian(Distribution):
    def __init__(
            self,
            dim,
            truncate_std=False,
            max_std=2.0,
            min_std=0.1,
            prior_mean=None,
            prior_stddev=None,
            prior_trainable=False,
            init_prior_mean=None,
            init_prior_stddev=None,
    ):
        Serializable.quick_init(self, locals())

        self._name = "%sD_Gaussian_id_%s" % (dim, G_IDX)
        global G_IDX
        G_IDX += 1

        self._dim = dim
        self._truncate_std = truncate_std
        self._max_std = max_std
        self._min_std = min_std
        if prior_mean is None:
            prior_mean = np.zeros((dim,))
        if prior_stddev is None:
            prior_stddev = np.ones((dim,))
        self._prior_trainable = prior_trainable
        if prior_trainable:
            self._init_prior_mean = tf.reshape(
                tf.cast(prior_mean if init_prior_mean is None else init_prior_mean, floatX), [1, -1])
            self._init_prior_stddev = tf.reshape(
                tf.cast(prior_stddev if init_prior_stddev is None else init_prior_stddev, floatX), [1, -1])

            prior_mean = tf.get_variable(
                "prior_mean_%s" % self._name,
                initializer=tf.constant(prior_mean),
                dtype=floatX,
            )
            # forget it untill we code it numerically more stable param
            # prior_stddev = tf.get_variable("prior_stddev_%s" % self._name, initializer=prior_mean)
        self._prior_mean = tf.reshape(tf.cast(prior_mean, floatX), [1, dim])
        self._prior_stddev = tf.reshape(tf.cast(prior_stddev, floatX), [1, dim])

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    def logli(self, x_var, dist_info):
        mean = dist_info["mean"]
        stddev = dist_info["stddev"]
        epsilon = (x_var - mean) / (stddev + TINY)
        return tf.reduce_sum(
            - 0.5 * np.log(2 * np.pi) - tf.log(stddev + TINY) - 0.5 * tf.square(epsilon),
            reduction_indices=1,
        )

    def prior_dist_info(self, batch_size):
        batch_size = int(batch_size)
        mean = tf.tile(self._prior_mean, [batch_size, 1])
        stddev = tf.tile(self._prior_stddev, [batch_size, 1])
        return dict(mean=mean, stddev=stddev)

    def init_prior_dist_info(self, batch_size):
        batch_size = int(batch_size)
        mean = tf.tile(self._init_prior_mean, [batch_size, 1])
        stddev = tf.tile(self._init_prior_stddev, [batch_size, 1])
        return dict(mean=mean, stddev=stddev)

    def nonreparam_logli(self, x_var, dist_info):
        return tf.zeros_like(x_var[:, 0])

    def kl(self, p, q):
        p_mean = p["mean"]
        p_stddev = p["stddev"]
        q_mean = q["mean"]
        q_stddev = q["stddev"]
        # means: (N*D)
        # std: (N*D)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) + ln(\sigma_2/\sigma_1)
        numerator = tf.square(p_mean - q_mean) + tf.square(p_stddev) - tf.square(q_stddev)
        denominator = 2. * tf.square(q_stddev)
        return tf.reduce_sum(
            numerator / (denominator + TINY) + tf.log(q_stddev + TINY) - tf.log(p_stddev + TINY),
            reduction_indices=1
        )

    def sample_logli(self, dist_info):
        mean = dist_info["mean"]
        stddev = dist_info["stddev"]
        epsilon = tf.random_normal(universal_int_shape(mean), seed=get_current_seed())
        out = mean + epsilon * stddev
        return out, self.logli(out, dist_info)

    @property
    def dist_info_keys(self):
        return ["mean", "stddev"]

    def activate_dist(self, flat_dist):
        mean = flat_dist[:, :self.dim]
        if self._truncate_std:
            stddev = tf.sqrt(tf.exp(flat_dist[:, self.dim:]))
            stddev = tf.clip_by_value(stddev, self._min_std, self._max_std)
        else:
            stddev = tf.sqrt(tf.exp(flat_dist[:, self.dim:]))
        return dict(mean=mean, stddev=stddev)

    def deactivate_dist(self, dict):
        return tf.concat(
            1,
            [dict["mean"], dict["stddev"]]
        )

    def entropy(self, dist_info):
        """
        :return: entropy for each minibatch entry
        """
        mu, std = dist_info["mean"], dist_info["stddev"]
        return tf.reduce_sum(
            0.5 * (np.log(2) + 2. * tf.log(std) + np.log(np.pi) + 1.),
            reduction_indices=[1]
        )


class Uniform(Gaussian):
    """
    This distribution will sample prior data from a uniform distribution, but
    the prior and posterior are still modeled as a Gaussian
    """

    def kl_prior(self):
        raise NotImplementedError

    # def prior_dist_info(self, batch_size):
    #     raise NotImplementedError

    # def logli_prior(self, x_var)
    #     #
    #     raise NotImplementedError

    def sample_prior(self, batch_size):
        return tf.random_uniform([batch_size, self.dim], minval=-1., maxval=1., seed=get_current_seed())


class SparseUniform(Gaussian):
    """
    This distribution will sample prior data from a uniform distribution, but
    the prior and posterior are still modeled as a Gaussian

    except the noise is masked out 50% of the time
    """

    def kl_prior(self):
        raise NotImplementedError

    def sample_prior(self, batch_size):
        shp = [batch_size, self.dim]
        return tf.random_uniform(
            shp,
            minval=-1., maxval=1.,
            seed=get_current_seed()
        ) * tf.select(
            0.5 >= tf.random_uniform(
                shp,
                minval=0., maxval=1.,
                seed=get_current_seed()
            ),
            tf.zeros(shp),
            tf.ones(shp)
        )


class Bernoulli(Distribution):
    def __init__(self, dim, smooth=None):
        Serializable.quick_init(self, locals())

        self._smooth = smooth
        self._dim = dim
        print("Bernoulli(dim=%s, smooth=%s)" % (dim, smooth))

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self._dim

    @property
    def dist_info_keys(self):
        return ["p"]

    def logli(self, x_var, dist_info):
        p = dist_info["p"]
        if self._smooth is not None:
            x_var = x_var * self._smooth
        return tf.reduce_sum(
            x_var * tf.log(p + TINY) + (1.0 - x_var) * tf.log(1.0 - p + TINY),
            reduction_indices=1
        )

    def entropy(self, dist_info):
        prob = dist_info["p"]
        neg_prob = 1. - prob
        return -tf.reduce_sum(prob * tf.log(prob + TINY), reduction_indices=1) \
               - tf.reduce_sum(neg_prob * tf.log(neg_prob + TINY), reduction_indices=1)

    def nonreparam_logli(self, x_var, dist_info):
        return self.logli(x_var, dist_info)

    def activate_dist(self, flat_dist):
        return dict(p=tf.nn.sigmoid(flat_dist))

    def sample_logli(self, dist_info):
        p = dist_info["p"]
        out = tf.cast(tf.less(tf.random_uniform(p.get_shape(), seed=get_current_seed()), p), tf.float32)
        return out, self.logli(out, dist_info)

    def prior_dist_info(self, batch_size):
        return dict(p=0.5 * tf.ones([batch_size, self.dim]))
    
class PiecewiseLinear(Distribution):
    # return [-1, 1) by piece-wise linear density in [-1,0) and [0,1)
    def __init__(
            self, shape,
    ):
        Serializable.quick_init(self, locals())

        self._shape = shape
        self._dim = np.prod(shape)

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    @property
    def dist_info_keys(self):
        return ["a1", "a2", "b"]

    def intercept(self, a1, a2):
        return (2. + a1 - a2) / 4

    def inverse_cdf(self, p, dist_info):
        a1, a2, b = dist_info["a1"], dist_info["a2"], dist_info["b"]
        kink_p = b - a1/2.
        b1 = b - a1
        p = tf.clip_by_value(p, 0., kink_p - 1e-2)
        # not yet numerically stable
        return tf.select(
            p < kink_p,
            tf.select(
                tf.abs(a1) > 1e-2,
                (-b1+tf.sqrt(b1**2.+2*a1*p+1e-8))/(a1+1e-8) - 1.,
                p/(b1 + 1e-8) - 1.,
                ),
            tf.select(
                tf.abs(a2) > 1e-2,
                (-b+tf.sqrt(b**2+2*a2*(p-kink_p)+1e-8))/(a2+1e-8),
                (p-kink_p)/(b+1e-8)
            )
        )

    def logli(self, x_var, dist_info):
        a1, a2, b = dist_info["a1"], dist_info["a2"], dist_info["b"]
        pdf = tf.select(
            x_var < 0.,
            b + a1 * x_var,
            b + a2 * x_var,
            )
        return tf.reduce_sum(
            tf.log(pdf + 1e-20),
            reduction_indices=[1]
        )

    def activate_dist(self, flat_dist):
        a1 = tf.tanh(flat_dist[:, :self.dim])
        a2 = tf.tanh(flat_dist[:, self.dim:])
        b = self.intercept(a1, a2)
        return dict(
            a1=a1, a2=a2, b=b
        )

    def sample_logli(self, dist_info):
        a1, a2, b = dist_info["a1"], dist_info["a2"], dist_info["b"]
        p = tf.random_uniform(
            shape=universal_int_shape(a1),
            maxval=1.,
        )
        samples = self.inverse_cdf(p, dist_info)
        logli = self.logli(samples, dist_info)
        samples = tf.Print(samples, [
            tf.check_numerics(samples,"samples"), tf.check_numerics(logli, "logli"),
            tf.gradients(logli, a1),
            tf.gradients(logli, a2),
            tf.gradients(logli, b)
        ])

        return tf.reshape(
            samples,
            [-1]+list(self._shape)
        ), logli

    def prior_dist_info(self, batch_size):
        return self.activate_dist(
            np.zeros([batch_size, self.dist_flat_dim])
        )

class Kumaraswamy(Distribution):
    # return [0, 1) by Kumuaraswamy(a,b)
    def __init__(
            self, shape,
    ):
        Serializable.quick_init(self, locals())

        self._shape = shape
        self._dim = np.prod(shape)

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    @property
    def dist_info_keys(self):
        return ["a", "b"]

    def inverse_cdf(self, p, dist_info):
        a, b = extract(dist_info, "a", "b")
        return (
            1. - (1. - p) ** (1. / b)
        ) ** (1. / a)

    def logli(self, x_var, dist_info):
        a, b = extract(dist_info, "a", "b")
        raw = tf.log(a) + tf.log(b) + (a-1.)*tf.log(x_var+1e-7) + (b-1.)*tf.log(1.-x_var**a+1e-7)

        return tf.reduce_sum(
            raw,
            reduction_indices=[1]
        )

    def activate_dist(self, flat_dist):
        # K(0.01, 0.01) can create extremely U-shaped distribution already
        a = tf.exp(flat_dist[:, :self.dim]) + .01
        b = tf.exp(flat_dist[:, self.dim:]) + .01
        return dict(
            a=a, b=b
        )

    def sample_logli(self, dist_info):
        a, b = extract(dist_info, "a", "b")
        p = tf.random_uniform(
            shape=universal_int_shape(a),
            maxval=1.,
        )
        # samples = self.inverse_cdf(p, dist_info)
        # samples = (
        #            1. - (1. - p) ** (1. / b)
        #        ) ** (1. / a)
        samples = tf.clip_by_value(
                      1. - (1. - p) ** (1. / b)
                      , 1e-7, 1
                  ) ** (1. / a)
        # with tf.device("/cpu:0"):
        #     with tf.control_dependencies([
        #         tf.assert_non_negative(samples),
        #         tf.assert_less_equal(samples, 1.),
        #         tf.Print(samples, [
        #             "min", tf.reduce_min(1. - (1. - p) ** (1. / b)),
        #             "max", tf.reduce_max(1. - (1. - p) ** (1. / b)),
        #         ])
        #     ]):
        #         samples = tf.identity(samples)
        raw = tf.log(a) + tf.log(b) + (a-1.)*tf.log(samples+1e-7) + (b-1.)*tf.log(
            (1. - p) ** (1. / b) + 1e-7
        )
        logli = tf.reduce_sum(raw, reduction_indices=[1])

        # logli = self.logli(samples, dist_info)
        # samples = tf.Print(samples, [
        #     tf.check_numerics(samples,"samples"), tf.check_numerics(logli, "logli"),
        #     tf.check_numerics(tf.gradients(logli, a), "fake grad a"),
        # ])

        INSPECT["kuma"].append(locals())
        return tf.reshape(
            samples,
            [-1]+list(self._shape)
        ), logli

    def prior_dist_info(self, batch_size):
        return self.activate_dist(
            np.zeros([batch_size, self.dist_flat_dim])
        )

THRESHOLD = 1e-3

class Logistic(Distribution):
    def __init__(
            self, shape,
            init_scale=1., init_mean=1.,
    ):
        Serializable.quick_init(self, locals())

        self._shape = shape
        self._dim = np.prod(shape)
        self._init_mean = init_mean
        self._init_scale = init_scale

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    @property
    def dist_info_keys(self):
        return ["mu", "scale"]

    def cdf(self, x_var, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        return tf.nn.sigmoid((x_var - mu) / (scale + 1e-7))

    def inverse_cdf(self, p, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        p = tf.clip_by_value(p, 1e-6, 1-1e-6)
        return mu + scale * (tf.log(p) - tf.log(1 - p))

    def logli(self, x_var, dist_info):
        x_var = tf.reshape(x_var, [-1, self.dim])
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        bs = int_shape(mu)[0]
        # untruncated logprob calculation
        neg_standardized = -(x_var - mu)/(scale + 1e-7)
        neg_standardized = neg_standardized
        raw_logli = neg_standardized - tf.log(scale + 1e-20) \
                    - 2. * tf.nn.softplus(neg_standardized)

        return tf.reduce_sum(raw_logli, reduction_indices=[1])

    def entropy(self, dist_info):
        raise NotImplemented

    def activate_dist(self, flat_dist):
        return dict(
            mu=self._init_mean * (flat_dist[:, :self.dim]),
            scale=self._init_scale * tf.exp(flat_dist[:, self.dim:]),
        )

    def sample_logli(self, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        bs = int_shape(mu)[0]
        p = tf.random_uniform(
            shape=universal_int_shape(mu),
            minval=1e-20,
            maxval=1. - 1e-20,
        )
        samples = self.inverse_cdf(p, dist_info)

        return tf.reshape(samples, [-1]+list(self._shape)), self.logli(samples, dist_info)

    def prior_dist_info(self, batch_size):
        return dict(
            mu=tf.zeros([batch_size, self._dim]),
            scale=tf.zeros([batch_size, self._dim]),
        )

class TruncatedLogistic(Distribution):
    def __init__(
            self, shape, lo, hi,
            init_scale=0.3, init_mean=2.,
            # init_scale=1, init_mean=1.,
    ):
        Serializable.quick_init(self, locals())

        self._shape = shape
        self._dim = np.prod(shape)
        assert hi > lo
        self._lo = (np.ones([1, self._dim]) * lo).astype(np.float32)
        self._hi = (np.ones([1, self._dim]) * hi).astype(np.float32)
        self._init_mean = init_mean
        self._init_scale = init_scale

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    @property
    def dist_info_keys(self):
        return ["mu", "scale"]

    def cdf(self, x_var, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        return tf.nn.sigmoid((x_var - mu) / (scale + 1e-7))

    def inverse_cdf(self, p, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        p = tf.clip_by_value(p, 1e-6, 1-1e-6)
        return mu + scale * (tf.log(p) - tf.log(1 - p))

    def logli(self, x_var, dist_info):
        x_var = tf.reshape(x_var, [-1, self.dim])
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        bs = int_shape(mu)[0]
        p_lo, p_hi = self.cdf(self._lo, dist_info), self.cdf(self._hi, dist_info)
        span = p_hi - p_lo
        # untruncated logprob calculation
        neg_standardized = -(x_var - mu)/(scale + 1e-7)
        neg_standardized = tf.check_numerics(neg_standardized, "neg_std")
        # neg_standardized = tf.Print(
        #     neg_standardized, [
        #         "neg_std_max", tf.reduce_max(neg_standardized),
        #         "scale_min", tf.reduce_min(scale),
        #         "mu_max", tf.reduce_max(mu),
        #     ]
        # )
        # log_exp_approx = tf.select(
        #     neg_standardized > 70.,
        #     neg_standardized,
        #     tf.select(
        #         neg_standardized < -10.,
        #         tf.exp(neg_standardized),
        #         tf.log(1. + tf.exp(neg_standardized))
        #     )
        # )
        log_exp_approx = tf.nn.softplus(neg_standardized)
        # log_exp_approx = tf.Print(
        #     log_exp_approx, [
        #         "span_min", tf.reduce_min(span),
        #         "softplux_approx_mu", tf.reduce_mean(log_exp_approx),
        #         "softplux_approx_min", tf.reduce_min(log_exp_approx),
        #         "softplux_approx_max", tf.reduce_max(log_exp_approx),
        #     ]
        # )
        raw_logli = neg_standardized - tf.log(scale + 1e-20) \
                    - 2. * log_exp_approx

        logli_out = tf.select(
            span > THRESHOLD,
            raw_logli - tf.log(
                tf.clip_by_value(span, 1e-7, 1.)
            ),
            tf.tile(-tf.log(self._hi - self._lo), [bs, 1])
        )
        return tf.reduce_sum(logli_out, reduction_indices=[1])

    def entropy(self, dist_info):
        raise NotImplemented

    def activate_dist(self, flat_dist):
        return dict(
            mu=self._init_mean * (flat_dist[:, :self.dim]),
            scale=self._init_scale * tf.exp(flat_dist[:, self.dim:]),
        )

    def sample_logli(self, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        bs = int_shape(mu)[0]
        p_lo, p_hi = self.cdf(self._lo, dist_info), self.cdf(self._hi, dist_info)
        span = p_hi - p_lo
        p_delta = tf.random_uniform(
            shape=universal_int_shape(mu),
            maxval=1.,
        ) * span
        samples = self.inverse_cdf(p_lo+p_delta, dist_info)

        # untruncated logprob calculation
        neg_standardized = -(samples - mu)/(scale + 1e-7)
        neg_standardized = tf.check_numerics(neg_standardized, "neg_std")
        log_exp_approx = tf.nn.softplus(neg_standardized)
        raw_logli = neg_standardized - tf.log(scale + 1e-20) \
                    - 2. * log_exp_approx

        span = tf.check_numerics(span, "cdf_span")
        samples_out = tf.select(
            span > THRESHOLD,
            samples,
            tf.random_uniform(
                shape=universal_int_shape(mu),
                minval=self._lo[0],
                maxval=self._hi[0],
            )
        )
        logli_out = tf.select(
            span > THRESHOLD,
            raw_logli - tf.log(
                tf.clip_by_value(span, 1e-7, 1.)
            ),
            tf.tile(-tf.log(self._hi - self._lo), [bs, 1])
        )
        # logli_out = tf.Print(
        #     logli_out,
        #     [
        #         # "p_hi", p_hi, "p_lo", p_lo,
        #         "spans", span, "samples", samples_out,
        #         "logli", logli_out,
        #         "neg_std", neg_standardized,
        #         "mu", mu,
        #         "scale", scale,
        #     ]
        # )

        return tf.reshape(samples_out, [-1]+list(self._shape)), tf.reduce_sum(logli_out, reduction_indices=[1])

    def prior_dist_info(self, batch_size):
        return dict(
            mu=np.zeros([batch_size, self._dim]),
            scale=np.ones([batch_size, self._dim]),
        )

class DiscretizedLogistic(Distribution):
    # assume to be -0.5 ~ 0.5
    def __init__(self, dim, bins=256., init_scale=0.1):
        Serializable.quick_init(self, locals())

        self._dim = dim
        self._bins = bins
        self._init_scale = init_scale

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    @property
    def dist_info_keys(self):
        return ["mu", "scale"]

    def cdf(self, x_var, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        return tf.nn.sigmoid((x_var - mu) / scale)

    def floor(self, x_var):
        floored = tf.floor(x_var * self._bins) / self._bins
        return floored

    def logli(self, x_var, dist_info):
        floored = self.floor(x_var)
        cdf_hi = self.cdf(floored + 1. / self._bins, dist_info)
        cdf_lo = self.cdf(floored, dist_info)
        cdf_diff = tf.select(
            floored <= -0.5 + 1. / self._bins - 1e-5,
            cdf_hi,
            tf.select(
                floored >= 0.5 - 1. / self._bins,
                1 - cdf_lo,
                cdf_hi - cdf_lo
            )
        )
        return tf.reduce_sum(
            tf.log(
                cdf_diff + 1e-7
            ),
            reduction_indices=[1],
        )

    def entropy(self, dist_info):
        # XXX fixme
        return 0.

    def nonreparam_logli(self, x_var, dist_info):
        return self.logli(x_var, dist_info)

    def activate_dist(self, flat_dist):
        return dict(
            mu=(flat_dist[:, :self.dim]),
            scale=tf.exp(flat_dist[:, self.dim:]) * self._init_scale,
        )

    def sample_logli(self, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        p = tf.random_uniform(
            shape=universal_int_shape(mu),
            minval=1e-5,
            maxval=1. - 1e-5,
        )
        real_logit = mu + scale * (tf.log(p) - tf.log(1 - p))  # inverse cdf according to wiki
        clipped = tf.clip_by_value(real_logit, -0.5, 0.5 - 1. / self._bins)
        out = self.floor(clipped)
        return out, self.logli(out, dist_info)

    def prior_dist_info(self, batch_size):
        return dict(
            mu=0.0 * np.ones([batch_size, self._dim]),
            scale=np.ones([batch_size, self._dim]) * self._init_scale,
        )


class MeanDiscretizedLogistic(DiscretizedLogistic):
    def sample_logli(self, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        out = mu
        return out, self.logli(out, dist_info)


class DiscretizedLogistic2(Distribution):
    # assume to be -0.5 ~ 0.5
    def __init__(self, dim, bins=256., init_scale=0.1):
        Serializable.quick_init(self, locals())

        self._dim = dim
        self._bins = bins
        self._init_scale = init_scale

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    @property
    def dist_info_keys(self):
        return ["mu", "scale"]

    def raw(self, x_var, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        raw = ((x_var - mu) / scale)
        return raw

    def cdf(self, x_var, dist_info):
        return tf.nn.sigmoid(self.raw(x_var, dist_info))

    def log_cdf(self, x_var, dist_info):
        raw = self.raw(x_var, dist_info)
        return raw - tf.nn.softplus(raw)

    # centered_x = x - means
    # inv_stdv = tf.exp(-log_scales)
    # plus_in = inv_stdv * (centered_x + 1./255.)
    # cdf_plus = tf.nn.sigmoid(plus_in)
    # min_in = inv_stdv * (centered_x - 1./255.)
    # cdf_min = tf.nn.sigmoid(min_in)
    # log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    # cdf_delta = cdf_plus - cdf_min
    # mid_in = inv_stdv * centered_x
    # log_pdf_mid = -mid_in - log_scales - 2.*tf.nn.softplus(-mid_in)
    # log_probs = tf.select(
    #     x < -0.999,
    #     log_cdf_plus,
    #     tf.select(
    #         x > 0.999,
    #         log_one_minus_cdf_min,
    #         tf.select(
    #             cdf_delta > 1e-3,
    #             tf.log(cdf_delta + 1e-7),
    #             log_pdf_mid - np.log(127.5)
    #         )
    #     )
    # )

    def logli(self, x_var, dist_info):
        x_lo = x_var - .5 / self._bins
        x_hi = x_var + .5 / self._bins
        cdf_diff = self.cdf(x_hi, dist_info) - \
                   self.cdf(x_lo, dist_info)
        log_cdf_x_hi = self.log_cdf(x_hi, dist_info)
        log_one_minus_cdf_x_lo = -tf.nn.softplus(self.raw(x_lo, dist_info))
        log_probs = tf.select(
            x_var < -0.5 + 1. / self._bins - 1e-5,
            log_cdf_x_hi,
            tf.select(
                x_var > 0.5 - 1. / self._bins,
                log_one_minus_cdf_x_lo,
                tf.select(
                    cdf_diff > 1e-3,
                    tf.log(cdf_diff + 1e-7),
                    tf.log(cdf_diff + 1e-7),
                    # log_pdf_mid - np.log(127.5)
                )
            )
        )
        return tf.reduce_sum(
            log_probs,
            reduction_indices=[1],
        )

    def entropy(self, dist_info):
        # XXX fixme
        return 0.

    def nonreparam_logli(self, x_var, dist_info):
        return self.logli(x_var, dist_info)

    def activate_dist(self, flat_dist):
        return dict(
            mu=(flat_dist[:, :self.dim]),
            scale=tf.exp(flat_dist[:, self.dim:]) * self._init_scale,
        )

    def sample(self, dist_info):
        mu = dist_info["mu"]
        scale = dist_info["scale"]
        p = tf.random_uniform(shape=mu.get_shape(), seed=get_current_seed())
        real_logit = mu + scale * (tf.log(p) - tf.log(1 - p))  # inverse cdf according to wiki
        clipped = tf.clip_by_value(real_logit, -0.5, 0.5 - 1. / self._bins)
        return (clipped)

    def prior_dist_info(self, batch_size):
        return dict(
            mu=0.0 * np.ones([batch_size, self._dim]),
            scale=np.ones([batch_size, self._dim]) * self._init_scale,
        )


class MeanBernoulli(Bernoulli):
    """
    Behaves almost the same as the usual Bernoulli distribution, except that when sampling from it, directly
    return the mean instead of sampling binary values
    """

    def sample(self, dist_info):
        return dist_info["p"]

    def nonreparam_logli(self, x_var, dist_info):
        return tf.zeros_like(x_var[:, 0])


class TanhMeanBernoulli(MeanBernoulli):
    """
    Behaves almost the same as the usual Bernoulli distribution, except that when sampling from it, directly
    return the mean instead of sampling binary values
    """

    def activate_dist(self, flat_dist):
        return dict(p=tf.nn.tanh(flat_dist))


# class MeanCenteredUniform(MeanBernoulli):
#     """
#     Behaves almost the same as the usual Bernoulli distribution, except that when sampling from it, directly
#     return the mean instead of sampling binary values
#     """


class Product(Distribution):
    def __init__(self, dists):
        """
        :type dists: list[Distribution]
        """
        Serializable.quick_init(self, locals())

        self._dists = dists

    @property
    def dists(self):
        return list(self._dists)

    @property
    def dim(self):
        return sum(x.dim for x in self.dists)

    @property
    def effective_dim(self):
        return sum(x.effective_dim for x in self.dists)

    @property
    def dims(self):
        return [x.dim for x in self.dists]

    @property
    def dist_flat_dims(self):
        return [x.dist_flat_dim for x in self.dists]

    @property
    def dist_flat_dim(self):
        return sum(x.dist_flat_dim for x in self.dists)

    @property
    def dist_info_keys(self):
        ret = []
        for idx, dist in enumerate(self.dists):
            for k in dist.dist_info_keys:
                ret.append("id_%d_%s" % (idx, k))
        return ret

    def split_dist_info(self, dist_info):
        ret = []
        for idx, dist in enumerate(self.dists):
            cur_dist_info = dict()
            for k in dist.dist_info_keys:
                cur_dist_info[k] = dist_info["id_%d_%s" % (idx, k)]
            ret.append(cur_dist_info)
        return ret

    def join_dist_infos(self, dist_infos):
        ret = dict()
        for idx, dist, dist_info_i in zip(itertools.count(), self.dists, dist_infos):
            for k in dist.dist_info_keys:
                ret["id_%d_%s" % (idx, k)] = dist_info_i[k]
        return ret

    def split_var(self, x):
        """
        Split the tensor variable or value into per component.
        """
        cum_dims = list(np.cumsum(self.dims))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.dists):
            sliced = x[:, slice_from:slice_to]
            out.append(sliced)
        return out

    def join_vars(self, xs):
        """
        Join the per component tensor variables into a whole tensor
        """
        return tf.concat(1, xs)

    def split_dist_flat(self, dist_flat):
        """
        Split flat dist info into per component
        """
        cum_dims = list(np.cumsum(self.dist_flat_dims))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.dists):
            sliced = dist_flat[:, slice_from:slice_to]
            out.append(sliced)
        return out

    def prior_dist_info(self, batch_size):
        ret = []
        for dist_i in self.dists:
            ret.append(dist_i.prior_dist_info(batch_size))
        return self.join_dist_infos(ret)

    def kl(self, p, q):
        ret = tf.constant(0.)
        for p_i, q_i, dist_i in zip(self.split_dist_info(p), self.split_dist_info(q), self.dists):
            ret += dist_i.kl(p_i, q_i)
        return ret

    def activate_dist(self, dist_flat):
        ret = dict()
        for idx, dist_flat_i, dist_i in zip(itertools.count(), self.split_dist_flat(dist_flat), self.dists):
            dist_info_i = dist_i.activate_dist(dist_flat_i)
            for k, v in dist_info_i.items():
                ret["id_%d_%s" % (idx, k)] = v
        return ret

    def sample(self, dist_info):
        ret = []
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret.append(tf.cast(dist_i.sample(dist_info_i), tf.float32))
        return tf.concat(1, ret)

    def sample_prior(self, batch_size):
        ret = []
        for dist_i in self.dists:
            ret.append(tf.cast(dist_i.sample_prior(batch_size), tf.float32))
        return tf.concat(1, ret)

    def logli(self, x_var, dist_info):
        ret = tf.constant(0.)
        for x_i, dist_info_i, dist_i in zip(self.split_var(x_var), self.split_dist_info(dist_info), self.dists):
            ret += dist_i.logli(x_i, dist_info_i)
        return ret

    def marginal_logli(self, x_var, dist_info):
        ret = tf.constant(0.)
        for x_i, dist_info_i, dist_i in zip(self.split_var(x_var), self.split_dist_info(dist_info), self.dists):
            ret += dist_i.marginal_logli(x_i, dist_info_i)
        return ret

    def entropy(self, dist_info):
        ret = tf.constant(0.)
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret += dist_i.entropy(dist_info_i)
        return ret

    def marginal_entropy(self, dist_info):
        ret = tf.constant(0.)
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret += dist_i.marginal_entropy(dist_info_i)
        return ret

    def nonreparam_logli(self, x_var, dist_info):
        ret = tf.constant(0.)
        for x_i, dist_info_i, dist_i in zip(self.split_var(x_var), self.split_dist_info(dist_info), self.dists):
            ret += dist_i.nonreparam_logli(x_i, dist_info_i)
        return ret


class Mixture(Distribution):
    def __init__(self, pairs, trainable=True):
        Serializable.quick_init(self, locals())

        assert len(pairs) >= 1
        self._pairs = pairs
        self._dim = pairs[0][0].dim
        self._dims = [dist.dist_flat_dim for dist, _ in pairs]
        self._dist_flat_dim = np.product(self._dims)
        for dist, p in pairs:
            assert self._dim == dist.dim

    def init_mode(self):
        for d, _ in self._pairs:
            d.init_mode()

    def train_mode(self):
        for d, _ in self._pairs:
            d.train_mode()

    def split(self, x):
        def go():
            i = 0
            for idim in self._dims:
                yield x[:, i:i + idim]
                i += idim

        return list(go())

    def merge(self, xs):
        return tf.concat(1, xs)

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return sum([
                       d.dist_flat_dim for d, p in self._pairs
                       ])

    @property
    def modes(self):
        return len(self._pairs)

    @property
    def effective_dim(self):
        return self._dim

    def logli(self, x, dist_info):
        infos = dist_info["infos"]
        # li = 0.
        loglips = []
        for pair, idist_info_flat in zip(self._pairs, infos):
            dist, p = pair
            # li += tf.exp(dist.logli(x, idist_info)) * p
            idist_info = dist.activate_dist(idist_info_flat)
            loglips.append(dist.logli(x, idist_info) + tf.log(p))
        variate = tf.reduce_max(loglips, reduction_indices=0, keep_dims=True)
        return tf.log(tf.reduce_sum(tf.exp(loglips - variate), reduction_indices=0, keep_dims=True) + TINY) + variate

    def mode(self, x, dist_info):
        infos = dist_info["infos"]
        # li = 0.
        loglips = []
        for pair, idist_info in zip(self._pairs, infos):
            dist, p = pair
            loglips.append(dist.logli(x, idist_info) + tf.log(p))
        maxis = tf.arg_max(loglips, 0)
        return maxis

    def mode_distances(self, x, dist_info):
        infos = dist_info["infos"]
        # li = 0.
        loglips = []
        for pair, idist_info in zip(self._pairs, infos):
            dist, p = pair
            loglips.append(dist.logli(x, idist_info) + tf.log(p))
        maxis = tf.arg_max(loglips, 0)
        return maxis

    def prior_dist_info(self, batch_size):
        return dict(infos=[
            dist.deactivate_dist(dist.prior_dist_info(batch_size))
            for dist, _ in self._pairs
            ])

    def init_prior_dist_info(self, batch_size):
        return dict(infos=[dist.init_prior_dist_info(batch_size) for dist, _ in self._pairs])

    def nonreparam_logli(self, x_var, dist_info):
        return tf.zeros_like(x_var[:, 0])

    def kl(self, p, q):
        raise NotImplemented

    def entropy(self, dist_info):
        # XXX
        return 0.

    def sample_logli(self, dist_info):
        infos = dist_info["infos"]
        samples = [
            pair[0].sample(
                pair[0].activate_dist(iflat)
            )
            for pair, iflat in zip(self._pairs, infos)
        ]
        shp = int_shape(samples[0])
        samples = [
            tf.reshape(itr, [shp[0], -1]) for itr in samples
        ]
        bs = shp[0]
        prob = np.asarray([[p for _, p in self._pairs]] * bs)
        # ids = tf.multinomial(tf.log(prob), num_samples=1, seed=get_current_seed())[:, 0]
        ids = tf.multinomial(tf.log(prob), num_samples=1, seed=get_current_seed())
        onehot_table = tf.constant(np.eye(len(self._pairs), dtype=np.float32))
        onehot = tf.nn.embedding_lookup(onehot_table, ids)
        # return onehot, tf.constant(0.) + samples, tf.reduce_sum(
        #     tf.reshape(onehot, [bs, len(infos), 1]) * tf.transpose(samples, [1, 0, 2]),
        #     reduction_indices=1
        # )
        out = tf.reduce_sum(
            tf.reshape(onehot, [bs, len(infos), 1]) * tf.transpose(samples, [1, 0, 2]),
            reduction_indices=1
        )
        return tf.reshape(out, shp), self.logli(out, dist_info)

    def sample_one_mode(self, dist_info, mode):
        infos = dist_info["infos"]
        return self._pairs[mode][0].sample(infos[mode])

    @property
    def dist_info_keys(self):
        return ["infos"]

    @property
    def dists(self):
        return [d for d, _ in self._pairs]

    def activate_dist(self, flat_dist):
        def go():
            i = 0
            for dist, _ in self._pairs:
                yield flat_dist[:, i:i + dist.dist_flat_dim]
                i += dist.dist_flat_dim

        return dict(infos=list(go()))

    def deactivate_dist(self, dict):
        return tf.concat(
            1,
            dict["infos"]
        )


dist_book = pt.bookkeeper_for_default_graph()


# should have renamed this to AF (autoregressive flow)
class AR(Distribution):
    def __init__(
            self,
            dim,
            base_dist,
            depth=2,
            neuron_ratio=4,
            reverse=True,
            nl=tf.nn.relu,
            data_init_wnorm=True,
            data_init_scale=0.1,
            linear_context=False,
            gating_context=False,
            share_context=False,
            var_scope=None,
            rank=None,
            img_shape=None,
            keepprob=1.,
            clip=False,
            squash=False,
            ar_channels=False,
            mean_only=False,
    ):
        Serializable.quick_init(self, locals())

        self._name = "%sD_AR_id_%s" % (dim, G_IDX)
        global G_IDX
        G_IDX += 1

        self._dim = dim
        self._base_dist = base_dist
        self._depth = depth
        self._reverse = reverse
        self._wnorm = data_init_wnorm
        self._data_init = data_init_wnorm
        self._data_init_scale = data_init_scale
        self._linear_context = linear_context
        self._gating_context = gating_context
        self._context_dim = 0
        self._share_context = share_context
        self._rank = rank
        self._clip = clip
        self._squash = squash
        self._iaf_template = pt.template("y", books=dist_book)
        self._mean_only = mean_only
        if linear_context:
            lin_con = pt.template("linear_context", books=dist_book)
            self._linear_context_dim = 2 * dim * neuron_ratio
            self._context_dim += self._linear_context_dim
        if gating_context:
            gate_con = pt.template("gating_context", books=dist_book)
            self._gating_context_dim = 2 * dim * neuron_ratio
            self._context_dim += self._gating_context_dim

        if img_shape is None:
            assert depth >= 1
            assert keepprob == 1.
            from prettytensor import UnboundVariable
            with pt.defaults_scope(
                    activation_fn=nl,
                    wnorm=data_init_wnorm,
                    custom_phase=UnboundVariable('custom_phase'),
                    init_scale=self._data_init_scale,
                    var_scope=var_scope,
                    rank=rank,
            ):
                for di in range(depth):
                    self._iaf_template = \
                        self._iaf_template.arfc(
                            2 * dim * neuron_ratio,
                            ngroups=dim,
                            zerodiagonal=di == 0,  # only blocking the first layer can stop data flow
                            prefix="arfc%s" % di,
                        )
                    if di == 0:
                        if gating_context:
                            self._iaf_template *= (gate_con + 1).apply(tf.nn.sigmoid)
                        if linear_context:
                            self._iaf_template += lin_con
                self._iaf_template = \
                    self._iaf_template. \
                        arfc(
                        dim * 2,
                        activation_fn=None,
                        ngroups=dim,
                        prefix="arfc_last",
                    ). \
                        reshape([-1, self._dim, 2]). \
                        apply(tf.transpose, [0, 2, 1]). \
                        reshape([-1, 2 * self._dim])
        else:
            cur = self._iaf_template.reshape([-1, ] + list(img_shape))
            img_chn = img_shape[2]
            nr_channels = 2 * img_chn * neuron_ratio
            filter_size = 3
            extra_nins = 1

            from prettytensor import UnboundVariable
            with pt.defaults_scope(
                    activation_fn=tf.nn.elu,
                    wnorm=True,
                    custom_phase=UnboundVariable('custom_phase'),
                    init_scale=0.1,
                    ar_channels=ar_channels,
                    pixel_bias=True,
                    var_scope=None,
            ):
                for di in range(depth):
                    if di == 0:
                        cur = \
                            cur.ar_conv2d_mod(
                                filter_size,
                                nr_channels,
                                zerodiagonal=True,
                                prefix="step0",
                            )
                        context_shp = [-1, ] + img_shape[:2] + [nr_channels]
                        if gating_context:
                            cur *= (gate_con + 1).apply(tf.nn.sigmoid).reshape(context_shp)
                        if linear_context:
                            cur += lin_con.reshape([-1]).reshape(context_shp)
                    else:
                        cur = \
                            resconv_v1_customconv(
                                "ar_conv2d_mod",
                                dict(
                                    zerodiagonal=False,
                                ),
                                cur,
                                filter_size,
                                nr_channels,
                                nin=False,
                                gating=True,
                                slow_gating=True,
                            )
                    for ninidx in range(extra_nins):
                        if ar_channels:
                            cur += 0.1 * cur.ar_conv2d_mod(
                                1,
                                nr_channels,
                                zerodiagonal=False,
                                prefix="nin_ex_%s" % ninidx,
                            )
                        else:
                            # backward compatibility for resumption
                            cur += 0.1 * cur.conv2d_mod(
                                1,
                                nr_channels,
                                prefix="nin_ex_%s" % ninidx,
                            )
                    cur = cur.custom_dropout(keepprob)
                self._iaf_template = \
                    cur.ar_conv2d_mod(
                        filter_size,
                        img_chn * 2,
                        zerodiagonal=False,
                        activation_fn=None,
                    ).reshape(
                        [-1, ] + img_shape + [2]
                    ).apply(
                        tf.transpose,
                        [0, 4, 1, 2, 3]
                    ).reshape([-1, np.prod(img_shape) * 2])

    @overrides
    def init_mode(self):
        if self._wnorm:
            self._custom_phase = CustomPhase.init
            self._base_dist.init_mode()

    @overrides
    def train_mode(self):
        if self._wnorm:
            self._custom_phase = CustomPhase.train
            self._base_dist.train_mode()

    @property
    def dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self._dim

    def infer(self, x_var, lin_con=None, gating_con=None):
        in_dict = dict(
            y=x_var,
            custom_phase=self._custom_phase,
        )
        if self._linear_context:
            assert lin_con is not None
            in_dict["linear_context"] = lin_con
        if self._gating_context:
            assert gating_con is not None
            in_dict["gating_context"] = gating_con
        flat_iaf = self._iaf_template.construct(
            **in_dict
        ).tensor
        iaf_mu, iaf_logstd = flat_iaf[:, :self._dim], flat_iaf[:, self._dim:]
        if self._clip:
            iaf_mu = tf.clip_by_value(
                iaf_mu,
                -5,
                5
            )
            iaf_logstd = tf.clip_by_value(
                iaf_logstd,
                -4,
                4,
            )
        if self._squash:
            iaf_mu = tf.tanh(iaf_mu) * 2.7
            iaf_logstd = tf.tanh(iaf_logstd) * 1.5
        if self._mean_only:
            # TODO: fixme! wasteful impl
            iaf_logstd = tf.zeros_like(iaf_mu)
        return iaf_mu, (iaf_logstd)

    def logli(self, x_var, dist_info):
        iaf_mu, iaf_logstd = self.infer(x_var)
        z = x_var / tf.exp(iaf_logstd) - iaf_mu
        if self._reverse:
            z = tf.reverse(z, [False, True])
        return self._base_dist.logli(z, dist_info) - \
               tf.reduce_sum(iaf_logstd, reduction_indices=1)

    def logli_init_prior(self, x_var):
        return self._base_dist.logli_init_prior(x_var)

    def prior_dist_info(self, batch_size):
        return self._base_dist.prior_dist_info(batch_size)

    def sample_logli(self, info):
        return self.sample_n(info=info)

    def sample_n(self, n=100, info=None):
        print("warning, ar sample invoked")
        try:
            z, logpz = self._base_dist.sample_n(n=n, info=info)
        except AttributeError:
            if info:
                z, logpz = self._base_dist.sample_logli(info)
            else:
                z = self._base_dist.sample_prior(batch_size=n)
                logpz = self._base_dist.logli_prior(z)
        if self._reverse:
            z = tf.reverse(z, [False, True])
        go = z  # place holder
        for i in range(self._dim):
            iaf_mu, iaf_logstd = self.infer(go)

            go = iaf_mu + tf.exp(iaf_logstd) * z
        return go, logpz - tf.reduce_sum(iaf_logstd, reduction_indices=1)

        # def accm(go, _):
        #     iaf_mu, iaf_logstd = self.infer(go)
        #     go = iaf_mu + tf.exp(iaf_logstd)*z
        #     return go
        # go = tf.foldl(
        #     fn=accm,
        #     elems=np.arange(self._dim, dtype=np.int32),
        #     initializer=z,
        # )
        # return go, 0. # fixme

    @property
    def dist_info_keys(self):
        return self._base_dist.dist_info_keys

    @property
    def dist_flat_dim(self):
        return self._base_dist.dist_flat_dim

    def activate_dist(self, flat):
        return self._base_dist.activate_dist(flat)

    def nonreparam_logli(self, x_var, dist_info):
        raise "not defined"


class IAR(AR):
    def sample_logli(self, dist_info):
        z, logpz = self._base_dist.sample_logli(dist_info=dist_info)
        if self._reverse:
            z = tf.reverse(z, [False, True])
        iaf_mu, iaf_logstd = self.infer(
            z,
            lin_con=dist_info.get("linear_context"),
            gating_con=dist_info.get("gating_context"),
        )
        go = iaf_mu + tf.exp(iaf_logstd) * z
        return go, logpz - tf.reduce_sum(iaf_logstd, reduction_indices=1)

    def logli(self, x_var, dist_info):
        print("warning, iar logli invoked")
        go = x_var  # place holder
        for i in range(self._dim):
            iaf_mu, iaf_logstd = self.infer(
                go,
                lin_con=dist_info.get("linear_context"),
                gating_con=dist_info.get("gating_context"),
            )
            go = iaf_mu + tf.exp(iaf_logstd) * x_var
        logpz = self._base_dist.logli(go, dist_info)
        return logpz - tf.reduce_sum(iaf_logstd, reduction_indices=1)

    def inserting_context(self):
        if self._share_context:
            if (isinstance(self._base_dist, AR)):
                assert self._base_dist._share_context
                return False
        return self._linear_context or self._gating_context
        # this is for sharing context version
        # keys = self._base_dist.dist_info_keys
        # return self._linear_context and ("linear_context" not in keys)

    @property
    def dist_info_keys(self):
        keys = self._base_dist.dist_info_keys
        if self.inserting_context():
            if self._linear_context:
                keys = keys + ["linear_context"]
            if self._gating_context:
                keys = keys + ["gating_context"]
        return keys

    @property
    def dist_flat_dim(self):
        if self.inserting_context():
            return self._base_dist.dist_flat_dim + self._context_dim
        else:
            return self._base_dist.dist_flat_dim

    def activate_dist(self, flat):
        if self.inserting_context():
            out = dict()
            if self._linear_context:
                lin_context = flat[:, :self._linear_context_dim]
                flat = flat[:, self._linear_context_dim:]
                out["linear_context"] = lin_context
            if self._gating_context:
                lin_context = flat[:, :self._gating_context_dim]
                flat = flat[:, self._gating_context_dim:]
                out["gating_context"] = lin_context
            out = dict(out, **self._base_dist.activate_dist(flat))
            return out
        else:
            return self._base_dist.activate_dist(flat)


# MADE that outputs \theta_i for p_\theta_i(x_i | x<i)
# where conditional p is specified by tgt_dist
class DistAR(Distribution):
    def __init__(
            self,
            dim,
            tgt_dist,
            depth=2,
            neuron_ratio=4,
            nl=tf.nn.relu,
            data_init_wnorm=True,
            data_init_scale=0.1,
            linear_context=False,
            gating_context=False,
            mul_gating=False,
            op_context=False,
            share_context=False,
            var_scope=None,
            rank=None,
    ):
        Serializable.quick_init(self, locals())

        self._name = "%sD_AR_id_%s" % (dim, G_IDX)
        global G_IDX
        G_IDX += 1

        assert isinstance(tgt_dist, Mixture)
        nom = len(tgt_dist._pairs)
        mode_flat_dim = tgt_dist._pairs[0][0].dist_flat_dim
        for d, p in tgt_dist._pairs:
            assert isinstance(d, Gaussian) or isinstance(d, DiscretizedLogistic)

        self._dim = dim
        self._tgt_dist = tgt_dist
        self._depth = depth
        self._wnorm = data_init_wnorm
        self._data_init = data_init_wnorm
        self._data_init_scale = data_init_scale
        self._linear_context = linear_context
        self._gating_context = gating_context
        self._op_context = op_context
        self._context_dim = 0
        self._share_context = share_context
        self._rank = rank

        self._iaf_template = pt.template("y", books=dist_book)
        if linear_context:
            lin_con = pt.template("linear_context", books=dist_book)
            self._linear_context_dim = 2 * dim * neuron_ratio
            self._context_dim += self._linear_context_dim
        if gating_context:
            gate_con = pt.template("gating_context", books=dist_book)
            self._gating_context_dim = 2 * dim * neuron_ratio
            self._context_dim += self._gating_context_dim
        if op_context:
            op_con = pt.template("op_context", books=dist_book)
            # [bs, dim * hid_dim]
            hid_dim = dim * 2 * neuron_ratio
            self._op_context_dim = dim * hid_dim
            self._context_dim += self._op_context_dim
            mask = get_linear_ar_mask(dim, hid_dim, zerodiagonal=True)
            mask = mask.reshape([1, dim, hid_dim])
            # self._iaf_template = self._iaf_template.reshape(
            #     [-1, 1, dim]
            # ).join(
            #     [op_con.reshape([-1, dim, hid_dim]).apply(tf.nn.tanh) * mask],
            #     join_function=lambda ts: tf.batch_matmul(*ts),
            # ).reshape(
            #     [-1, hid_dim]
            # )
            self._iaf_template = (
                # op_con.reshape([-1, dim, hid_dim]).apply(tf.nn.tanh) * mask
                op_con.reshape([-1, dim, hid_dim]).apply(tf.nn.sigmoid) * mask
                # op_con.reshape([-1, dim, hid_dim]) * mask
            ).apply(
                tf.transpose,
                [0, 2, 1]
            ).join(
                [self._iaf_template.reshape([-1, dim, 1])],
                join_function=lambda ts: tf.batch_matmul(*ts),
            ).reshape(
                [-1, hid_dim]
            )

        assert depth >= 0
        from prettytensor import UnboundVariable
        with pt.defaults_scope(
                activation_fn=nl,
                wnorm=data_init_wnorm,
                custom_phase=UnboundVariable('custom_phase'),
                init_scale=self._data_init_scale,
                var_scope=var_scope,
                rank=rank,
        ):
            for di in range(depth):
                self._iaf_template = \
                    self._iaf_template.arfc(
                        2 * dim * neuron_ratio,
                        ngroups=dim,
                        zerodiagonal=di == 0 and (not op_context),  # only blocking the first layer can stop data flow
                        prefix="arfc%s" % di,
                    )
                if di == 0:
                    if gating_context:
                        if mul_gating:
                            self._iaf_template *= (gate_con)
                        else:
                            self._iaf_template *= (gate_con + 1).apply(tf.nn.sigmoid)
                    if linear_context:
                        self._iaf_template += lin_con
            # shortcut of [-1, dim, nom, 2]
            # -> [-1, nom, 2, dim]
            self._iaf_template = \
                self._iaf_template. \
                    arfc(
                    dim * 2 * nom,
                    activation_fn=None,
                    ngroups=dim,
                    prefix="arfc_last",
                ). \
                    reshape([-1, self._dim, 2 * nom]). \
                    apply(tf.transpose, [0, 2, 1]). \
                    reshape([-1, 2 * self._dim * nom])

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init
        # self._base_dist.init_mode()

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train
        # self._base_dist.train_mode()

    @property
    def dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self._dim

    def infer(self, x_var, lin_con=None, gating_con=None, op_con=None):
        in_dict = dict(
            y=x_var,
            custom_phase=self._custom_phase,
        )
        if self._linear_context:
            assert lin_con is not None
            in_dict["linear_context"] = lin_con
        if self._gating_context:
            assert gating_con is not None
            in_dict["gating_context"] = gating_con
        if self._op_context:
            assert op_con is not None
            in_dict["op_context"] = op_con
        flat_iaf = self._iaf_template.construct(
            **in_dict
        ).tensor
        return flat_iaf

    def logli(self, x_var, dist_info):
        tgt_flat = self.infer(
            x_var,
            lin_con=dist_info.get("linear_context"),
            gating_con=dist_info.get("gating_context"),
            op_con=dist_info.get("op_context"),
        )
        tgt_info = self._tgt_dist.activate_dist(tgt_flat)
        return self._tgt_dist.logli(x_var, tgt_info)

    def prior_dist_info(self, batch_size):
        return dict(batch_size=batch_size)

    def sample_logli(self, info):
        print("warning, dist_ar sample invoked")
        if "batch_size" not in info:
            bs = universal_int_shape((list(info.values())[0]))[0]
        else:
            bs = info["batch_size"]
        go = tf.zeros([bs, self.dim])  # place holder
        # HACK
        # go = tf.zeros([131072, self.dim]) # place holder
        for i in range(self._dim):
            tgt_flat = self.infer(
                go,
                lin_con=info.get("linear_context"),
                gating_con=info.get("gating_context"),
                op_con=info.get("op_context"),
            )
            tgt_info = self._tgt_dist.activate_dist(tgt_flat)
            proposed = self._tgt_dist.sample(tgt_info)
            go = tf.concat(
                1,
                [go[:, :i], proposed[:, i:]]
            )
        return go, self.logli(go, info)

    @property
    def dist_info_keys(self):
        keys = []
        if self._linear_context:
            keys = keys + ["linear_context"]
        if self._gating_context:
            keys = keys + ["gating_context"]
        if self._op_context:
            keys = keys + ["op_context"]
        return keys

    @property
    def dist_flat_dim(self):
        return self._context_dim

    def activate_dist(self, flat):
        out = dict()
        if self._linear_context:
            lin_context = flat[:, :self._linear_context_dim]
            flat = flat[:, self._linear_context_dim:]
            out["linear_context"] = lin_context
        if self._gating_context:
            lin_context = flat[:, :self._gating_context_dim]
            flat = flat[:, self._gating_context_dim:]
            out["gating_context"] = lin_context
        if self._op_context:
            op_context = flat[:, :self._op_context_dim]
            flat = flat[:, self._op_context_dim:]
            out["op_context"] = op_context
        return out

    def nonreparam_logli(self, x_var, dist_info):
        raise "not defined"


class ConvAR(Distribution):
    """Basic masked conv ar"""

    def __init__(
            self,
            tgt_dist,
            shape=(32, 32, 3),
            filter_size=3,
            depth=5,
            nr_channels=32,
            block="resnet",
            pixel_bias=False,
            context_dim=None,
            masked=True,
            nin=False,
            sanity=False,
            sanity2=False,
            tieweight=False,
            extra_nins=0,
            inp_keepprob=1.,
            legacy=False,
            concat_elu=False,
    ):
        Serializable.quick_init(self, locals())

        self._name = "%sD_ConvAR_id_%s" % (shape, G_IDX)
        global G_IDX
        G_IDX += 1

        self._tgt_dist = tgt_dist
        if tgt_dist is None:
            nr_mix = 10
            out_chn = 10 * nr_mix
        else:
            out_chn = tgt_dist.dist_flat_dim
        self._shape = shape
        self._dim = int(np.prod(shape))
        self._sanity = sanity
        self._sanity2 = sanity2
        context = context_dim is not None
        self._context = context
        self._context_dim = context_dim
        inp = pt.template("y", books=dist_book).reshape([-1, ] + list(shape))
        inp = inp.custom_dropout(inp_keepprob)
        if not legacy:
            self._inp_mask = tf.Variable(
                initial_value=0. if sanity else 1.,
                trainable=False,
                name="%s_inp_mask" % G_IDX,
                dtype=tf.float32,
            )

        if context:
            context_inp = \
                pt.template("context", books=dist_book). \
                    reshape([-1, ] + list(shape[:-1]) + [context_dim])
            if not legacy:
                self._context_mask = tf.Variable(
                    initial_value=0. if sanity2 else 1.,
                    trainable=False,
                    name="%s_context_mask" % G_IDX,
                    dtype=tf.float32,
                )
        self._custom_phase = CustomPhase.init

        from prettytensor import UnboundVariable
        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
        with pt.defaults_scope(
                activation_fn=nn.concat_elu if concat_elu else tf.nn.elu,
                wnorm=True,
                custom_phase=UnboundVariable('custom_phase'),
                init_scale=0.1,
                ar_channels=False,
                pixel_bias=pixel_bias,
                var_scope="ConvAR" if tieweight else None,
        ):
            if not legacy:
                inp = inp.mul_init_ensured(self._inp_mask)
            if context:
                if not legacy:
                    context_inp = context_inp.mul_init_ensured(self._context_mask)
                inp = inp.join(
                    [context_inp],
                )
            cur = inp
            peep_inp = inp.left_shift(filter_size - 1).down_shift()

            if masked:
                for di in range(depth):
                    if di == 0:
                        cur = \
                            cur.ar_conv2d_mod(
                                filter_size,
                                nr_channels,
                                zerodiagonal=di == 0,
                                prefix="step0",
                            )
                    else:
                        if "resnet" in block:
                            cur = \
                                resconv_v1_customconv(
                                    "ar_conv2d_mod",
                                    dict(zerodiagonal=False),
                                    cur,
                                    filter_size,
                                    nr_channels,
                                    nin=nin,
                                    gating="gat" in block,
                                )
                        elif block == "plstm":
                            use_peep = di % 2 == 1
                            cur, cur_nl = plstmconv_v1(
                                cur,
                                peep_inp if use_peep else inp,
                                filter_size, nr_channels,
                                prefix="peep" if use_peep else "ori",
                                op="ar_conv2d_mod",
                                args=dict(zerodiagonal=False),
                                args1=dict(
                                    zerodiagonal=not use_peep,
                                ),
                            )
                            if nin:
                                cur = cur + 0.1 * cur.conv2d_mod(
                                    1,
                                    nr_channels,
                                    prefix="nin",
                                )
                        else:
                            raise Exception("what")
                    for ninidx in range(extra_nins):
                        cur = cur + 0.1 * cur.conv2d_mod(
                            1,
                            nr_channels,
                            prefix="nin_ex_%s" % ninidx,
                        )

                self._iaf_template = \
                    cur.ar_conv2d_mod(
                        filter_size,
                        out_chn,
                        activation_fn=None,
                    )
            else:
                u_filter = (filter_size - 1) // 2
                upper = cur
                row = cur
                for di in range(depth):
                    upper = upper.conv2d_mod(
                        [u_filter, filter_size],
                        nr_channels,
                    ).down_shift(
                        size=u_filter
                    ) + upper
                    row = (
                        row + upper
                    ).conv2d_mod(
                        [1, u_filter],
                        nr_channels,
                    ).right_shift(
                        size=u_filter
                    )
                self._iaf_template = \
                    row.conv2d_mod(
                        1,
                        out_chn,
                        activation_fn=None,
                    )

    @overrides
    def init_mode(self):
        if self._tgt_dist:
            self._tgt_dist.init_mode()
        self._custom_phase = CustomPhase.init

    @overrides
    def train_mode(self):
        if self._tgt_dist:
            self._tgt_dist.train_mode()
        self._custom_phase = CustomPhase.train

    @property
    def dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self.dim

    def infer(self, x_var, context=None):
        in_dict = dict(
            y=x_var,
            custom_phase=self._custom_phase,
        )
        if self._context:
            in_dict["context"] = context
        conv_iaf = self._iaf_template.construct(
            **in_dict
        ).tensor
        # return self._tgt_dist.activate_dist(
        #     tf.reshape(conv_iaf, [-1, self._tgt_dist.dist_flat_dim])
        # )
        return conv_iaf

    def logli(self, x_var, info):
        raw = self.infer(x_var, info.get("context"))
        if self._tgt_dist:
            tgt_dict = self._tgt_dist.activate_dist(
                tf.reshape(raw, [-1, self._tgt_dist.dist_flat_dim])
            )

            flatten_loglis = self._tgt_dist.logli(
                tf.reshape(x_var, [-1, self._tgt_dist.dim]),
                tgt_dict
            )
        else:
            import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
            x_var = tf.reshape(
                x_var,
                [-1, ] + list(self._shape)
            )
            flatten_loglis = nn.discretized_mix_logistic(x_var * 2., raw)
        return tf.reduce_sum(
            tf.reshape(flatten_loglis, [-1, self._shape[0] * self._shape[1]]),
            reduction_indices=1
        )

    def prior_dist_info(self, batch_size):
        return {}

    def sample_logli(self, info):
        return self.sample_n(info=info)

    def reshaped_sample_logli(self, info):
        go, logpz = self._tgt_dist.sample_logli(info)
        go = tf.reshape(
            go,
            [-1] + list(self._shape)
        )
        return go, logpz

    def sample_n(self, n=100, info=None):
        print("warning, conv ar sample invoked")
        # if info is None:
        context = info.get("context")
        if context is not None:
            n = int_shape(context)[0]
        tgt_info = self._tgt_dist.prior_dist_info(n * self._shape[0] * self._shape[1])
        init, logpz = self.reshaped_sample_logli(tgt_info)
        go = init
        for i in range(self._dim):
            tgt_dict = self.infer(go, context)
            go, logpz = self.reshaped_sample_logli(tgt_dict)
        return go, logpz

        # go = tf.foldl(
        #     fn=lambda go, _: self.reshaped_sample_logli(
        #         self.infer(go, context)
        #     )[0],
        #     elems=np.arange(self._dim, dtype=np.int32),
        #     initializer=init,
        #     back_prop=False,
        # )
        # return go, 0. # fixme

    import functools
    @functools.lru_cache(maxsize=None)
    def sample_prior_sym(self, n):
        tgt_info = self._tgt_dist.prior_dist_info(n * self._shape[0] * self._shape[1])
        init, logpz = self.reshaped_sample_logli(tgt_info)
        return init

    @functools.lru_cache(maxsize=None)
    def infer_sym(self, n):
        x_var, context_var = \
            tf.placeholder(tf.float32, shape=[n, ] + list(self._shape)), \
            tf.placeholder(tf.float32, shape=[n, self.dist_flat_dim])
        tgt_dict = self._tgt_dist.activate_dist(self.infer(x_var, context_var))
        go, logpz = self.reshaped_sample_logli(tgt_dict)
        return x_var, context_var, go, tgt_dict

    def sample_dynamic(self, sess, info):
        print("warning, conv ar sample invoked")
        context = info.get("context")
        n = context.shape[0]
        go = sess.run(self.sample_prior_sym(n))
        x_var, context_var, go_sym, tgt_dict = self.infer_sym(n)

        raise "problem"
        pbar = ProgressBar(maxval=self._dim, )
        pbar.start()
        for i in range(self._dim):
            go = sess.run(
                go_sym,
                {
                    x_var: go,
                    context_var: context
                }
            )
            pbar.update(i)
        return go

    @property
    def dist_info_keys(self):
        return ["context"] if self._context else []

    @property
    def dist_flat_dim(self):
        return self._shape[0] * self._shape[1] * self._context_dim

    def activate_dist(self, flat):
        return dict(context=flat)

    def nonreparam_logli(self, x_var, dist_info):
        raise "not defined"


class PixelCNN(Distribution):
    """Porting tim's pixelcnn"""

    def __init__(
            self,
            shape=(32, 32, 3),
            nr_resnets=(5, 5, 5),
            nr_filters=64,
            nr_logistic_mix=10,
            nr_extra_nins=10,
            square=False,
            no_downpass=False,
            no_vgrowth=False,
    ):
        Serializable.quick_init(self, locals())

        self._name = "%sD_PixelCNN_id_%s" % (shape, G_IDX)
        self._shape = shape
        self._dim = np.prod(shape)
        global G_IDX
        G_IDX += 1
        self._custom_phase = CustomPhase.train
        self.infer_temp = tf.make_template(
            "infer",
            self.infer,
        )

        self.nr_resnets = nr_resnets
        self.nr_filters = nr_filters
        self.nr_logistic_mix = nr_logistic_mix
        self.nr_extra_nins = nr_extra_nins
        self.square = square
        self.no_downpass = no_downpass
        self.no_vgrowth = no_vgrowth

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train

    @property
    def dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self.dim

    def infer(self, x, context=None):
        import sandbox.pchen.InfoGAN.infogan.misc.imported.scopes as scopes
        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn

        def extra_nin(x):
            for _ in range(self.nr_extra_nins):
                x = nn.gated_resnet(x, conv=nn.nin)
            return x

        counters = {}
        with scopes.arg_scope(
                [nn.down_shifted_conv2d, nn.down_right_shifted_conv2d, nn.down_shifted_deconv2d,
                 nn.down_right_shifted_deconv2d, nn.nin],
                counters=counters, init=self._custom_phase == CustomPhase.init, ema=None
        ):

            # ////////// up pass ////////
            xs = nn.int_shape(x)
            x_pad = tf.concat(3, [x, tf.ones(
                xs[:-1] + [1])])  # add channel of ones to distinguish image from padding later on
            u_list = [nn.down_shifted_conv2d(x_pad, num_filters=self.nr_filters,
                                             filter_size=[2, 3])]  # stream for current row + up
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=self.nr_filters, filter_size=[1, 3])) + \
                       nn.right_shift(nn.down_right_shifted_conv2d(x, num_filters=self.nr_filters, filter_size=[2,
                                                                                                                1]))]  # stream for up and to the left

            for rep in range(self.nr_resnets[0]):
                if not self.no_vgrowth:
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(
                    nn.aux_gated_resnet(ul_list[-1], nn.down_shift(u_list[-1]), conv=nn.down_right_shifted_conv2d))

            for nr_resnet in self.nr_resnets[1:]:
                if not self.no_vgrowth:
                    u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=self.nr_filters, stride=[2, 2]))
                ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=self.nr_filters, stride=[2, 2]))

                for rep in range(nr_resnet):
                    if not self.no_vgrowth:
                        u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(
                        nn.aux_gated_resnet(ul_list[-1], nn.down_shift(u_list[-1]), conv=nn.down_right_shifted_conv2d))

            # /////// down pass ////////
            u = u_list.pop()
            ul = ul_list.pop()
            if not self.no_downpass:
                for idx, nr_resnet in enumerate(self.nr_resnets[:0:-1]):
                    for rep in range(nr_resnet + (0 if idx == 0 else 1)):
                        u = nn.aux_gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                        ul = nn.aux_gated_resnet(ul, tf.concat(3, [nn.down_shift(u), ul_list.pop()]),
                                                 conv=nn.down_right_shifted_conv2d)

                    u = nn.down_shifted_deconv2d(u, num_filters=self.nr_filters, stride=[2, 2])
                    u = extra_nin(u)
                    ul = nn.down_right_shifted_deconv2d(ul, num_filters=self.nr_filters, stride=[2, 2])
                    ul = extra_nin(ul)

                for rep in range(self.nr_resnets[0] + (1 if len(self.nr_resnets) > 1 else 0)):
                    u = nn.aux_gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                    u = extra_nin(u)
                ul = nn.aux_gated_resnet(ul, tf.concat(3, [nn.down_shift(u), ul_list.pop()]),
                                         conv=nn.down_right_shifted_conv2d)
                ul = extra_nin(ul)

                assert len(u_list) == 0
                assert len(ul_list) == 0

            x_out = nn.nin(nn.concat_elu(ul), 10 * self.nr_logistic_mix)

        return x_out

    def logli(self, x_var, info, spatial=False):
        import sandbox.pchen.InfoGAN.infogan.misc.imported.scopes as scopes
        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
        x_var = tf.reshape(
            x_var,
            [-1, ] + list(self._shape)
        ) * 2  # assumed to be [-1, 1]

        tgt_vec = self.infer_temp(x_var, info.get("context"))
        logli = nn.discretized_mix_logistic(
            x_var,
            tgt_vec
        )
        if spatial:
            return tf.reshape(logli, [-1, ] + list(self._shape[:2]))
        else:
            return tf.reduce_sum(
                tf.reshape(logli, [-1, self._shape[0] * self._shape[1]]),
                reduction_indices=1
            )

    def prior_dist_info(self, batch_size):
        return {}

    def sample_logli(self, info):
        raise NotImplemented

    @property
    def dist_info_keys(self):
        return ["context"] if self._context else []

    @property
    def dist_flat_dim(self):
        return self._shape[0] * self._shape[1] * self._context_dim

    def activate_dist(self, flat):
        return dict(context=flat)


class CondPixelCNN(Distribution):
    """conditional version with activations sharing"""

    def __init__(
            self,
            shape=(32, 32, 3),
            nr_resnets=(5, 5, 5),
            nr_filters=64,
            nr_cond_nins=1,
            nr_logistic_mix=10,
            nr_extra_nins=0,  # when this is a list, use repetively gated arch
            extra_compute=False,
            grayscale=False,
            no_downpass=False,
            no_vgrowth=False,
    ):
        Serializable.quick_init(self, locals())

        self._name = "%sD_PixelCNN_id_%s" % (shape, G_IDX)
        self._shape = shape
        self._dim = np.prod(shape)
        self._context_dim = nr_filters
        global G_IDX
        G_IDX += 1
        self._custom_phase = CustomPhase.train
        if isinstance(nr_extra_nins, list):
            self.infer_temp = tf.make_template(
                "infer",
                self.infer_rep,
            )
        else:
            self.infer_temp = tf.make_template(
                "infer",
                self.infer,
            )
        self.cond_temp = tf.make_template(
            "cond",
            self.cond,
        )

        self.nr_resnets = nr_resnets
        self.nr_filters = nr_filters
        self.nr_logistic_mix = nr_logistic_mix
        self.nr_cond_nins = nr_cond_nins
        self.nr_extra_nins = nr_extra_nins
        self.extra_compute = extra_compute
        self.grayscale = grayscale
        self.no_downpass = no_downpass
        self.no_vgrowth = no_vgrowth

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train

    @property
    def dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self.dim

    def infer(self, x, context=None):
        import sandbox.pchen.InfoGAN.infogan.misc.imported.scopes as scopes
        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
        assert not self.grayscale
        assert not self.no_downpass
        assert not self.no_vgrowth
        x = tf.reshape(
            x,
            [-1, ] + list(self._shape)
        )

        def extra_nin(x):
            for _ in range(self.nr_extra_nins):
                x = nn.gated_resnet(x, conv=nn.nin)
            return x

        counters = {}
        with scopes.arg_scope(
                [nn.down_shifted_conv2d, nn.down_right_shifted_conv2d, nn.down_shifted_deconv2d,
                 nn.down_right_shifted_deconv2d, nn.nin],
                counters=counters, init=self._custom_phase == CustomPhase.init, ema=None
        ):

            # ////////// up pass ////////
            xs = nn.int_shape(x)
            x_pad = tf.concat(3, [x, tf.ones(
                xs[:-1] + [1])])  # add channel of ones to distinguish image from padding later on
            u_list = [nn.down_shifted_conv2d(x_pad, num_filters=self.nr_filters,
                                             filter_size=[2, 3])]  # stream for current row + up
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=self.nr_filters, filter_size=[1, 3])) + \
                       nn.right_shift(nn.down_right_shifted_conv2d(x, num_filters=self.nr_filters, filter_size=[2,
                                                                                                                1]))]  # stream for up and to the left

            for rep in range(self.nr_resnets[0]):
                u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                if self.extra_compute:
                    u_list[-1] = extra_nin(u_list[-1])
                ul_list.append(
                    nn.aux_gated_resnet(ul_list[-1], nn.down_shift(u_list[-1]), conv=nn.down_right_shifted_conv2d))
                if self.extra_compute:
                    ul_list[-1] = extra_nin(ul_list[-1])

            for nr_resnet in self.nr_resnets[1:]:
                u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=self.nr_filters, stride=[2, 2]))
                ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=self.nr_filters, stride=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(
                        nn.aux_gated_resnet(ul_list[-1], nn.down_shift(u_list[-1]), conv=nn.down_right_shifted_conv2d))

            # /////// down pass ////////
            u = u_list.pop()
            ul = ul_list.pop()

            for idx, nr_resnet in enumerate(self.nr_resnets[:0:-1]):
                for rep in range(nr_resnet + (0 if idx == 0 else 1)):
                    u = nn.aux_gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                    u = extra_nin(u)
                    ul = nn.aux_gated_resnet(ul, tf.concat(3, [nn.down_shift(u), ul_list.pop()]),
                                             conv=nn.down_right_shifted_conv2d)
                    ul = extra_nin(ul)

                u = nn.down_shifted_deconv2d(u, num_filters=self.nr_filters, stride=[2, 2])
                ul = nn.down_right_shifted_deconv2d(ul, num_filters=self.nr_filters, stride=[2, 2])

            for rep in range(self.nr_resnets[0] + (1 if len(self.nr_resnets) > 1 else 0)):
                u = nn.aux_gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                u = extra_nin(u)
                ul = nn.aux_gated_resnet(ul, tf.concat(3, [nn.down_shift(u), ul_list.pop()]),
                                         conv=nn.down_right_shifted_conv2d)
                ul = extra_nin(ul)

            x_out = nn.nin(nn.concat_elu(ul), self.nr_filters)

        assert len(u_list) == 0
        assert len(ul_list) == 0

        return x_out

    def infer_rep(self, x, context=None):
        import sandbox.pchen.InfoGAN.infogan.misc.imported.scopes as scopes
        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
        x = tf.reshape(
            x,
            [-1, ] + list(self._shape)
        )
        if self.grayscale:
            r, g, b = x[:, :, :, 0:1], x[:, :, :, 1:2], x[:, :, :, 2:3]
            x = (0.299 * r) + (0.587 * g) + (0.114 * b)
        counters = {}

        def extra_nin(x, extra):
            for _ in range(extra):
                x = nn.gated_resnet(x, conv=nn.nin)
            return x

        # print("pixelcnn_context: %s" % context)
        # if context is not None:
        #     with scopes.arg_scope(
        #             [nn.down_shifted_conv2d, nn.down_right_shifted_conv2d, nn.down_shifted_deconv2d, nn.down_right_shifted_deconv2d, nn.nin],
        #             counters=counters, init=self._custom_phase == CustomPhase.init, ema=None,
        #     ):
        #         context = nn.nin(context, self.nr_filters*2)

        with scopes.arg_scope(
                [
                    nn.down_shifted_conv2d, nn.down_right_shifted_conv2d,
                    nn.down_shifted_deconv2d, nn.down_right_shifted_deconv2d,
                    nn.nin, nn.gated_resnet, nn.aux_gated_resnet
                ],
                counters=counters, init=self._custom_phase == CustomPhase.init, ema=None,
                context=context,
        ):

            # ////////// up pass ////////
            xs = nn.int_shape(x)
            x_pad = tf.concat(3, [x, tf.ones(
                xs[:-1] + [1])])  # add channel of ones to distinguish image from padding later on

            u_lists = {}
            ul_lists = {}
            for idx, extra in enumerate(self.nr_extra_nins):
                u_lists[idx] = [nn.down_shifted_conv2d(x_pad, num_filters=self.nr_filters,
                                                       filter_size=[2, 3])]  # stream for current row + up
                ul_lists[idx] = [
                    nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=self.nr_filters, filter_size=[1, 3])) + \
                    nn.right_shift(nn.down_right_shifted_conv2d(x, num_filters=self.nr_filters,
                                                                filter_size=[2, 1]))]  # stream for up and to the left

            for rep in range(self.nr_resnets[0]):
                for idx, extra in enumerate(self.nr_extra_nins):
                    if idx == 0:
                        if not self.no_vgrowth:
                            u_lists[idx].append(nn.gated_resnet(u_lists[idx][-1], conv=nn.down_shifted_conv2d))
                        assert not self.extra_compute
                        ul_lists[idx].append(nn.aux_gated_resnet(ul_lists[idx][-1], nn.down_shift(u_lists[idx][-1]),
                                                                 conv=nn.down_right_shifted_conv2d))
                    else:
                        if not self.no_vgrowth:
                            u_lists[idx].append(
                                nn.aux_gated_resnet(u_lists[idx][-1], u_lists[idx - 1][-1],
                                                    conv=nn.down_right_shifted_conv2d)
                            )
                        ul_lists[idx].append(
                            nn.aux_gated_resnet(
                                ul_lists[idx][-1],
                                tf.concat(3, [nn.down_shift(u_lists[idx][-1]), ul_lists[idx - 1][-1]]),
                                conv=nn.down_right_shifted_conv2d
                            )
                        )
            assert len(self.nr_resnets) == 1

            # /////// down pass ////////
            us = [u_lists[idx].pop() for idx in range(len(self.nr_extra_nins))]
            uls = [ul_lists[idx].pop() for idx in range(len(self.nr_extra_nins))]

            if not self.no_downpass:
                for rep in range(self.nr_resnets[0] + (1 if len(self.nr_resnets) > 1 else 0)):
                    for idx, extra in enumerate(self.nr_extra_nins):
                        if idx == 0:
                            us[idx] = nn.aux_gated_resnet(us[idx], u_lists[idx].pop(), conv=nn.down_shifted_conv2d)
                            us[idx] = extra_nin(us[idx], extra)
                            uls[idx] = nn.aux_gated_resnet(uls[idx],
                                                           tf.concat(3, [nn.down_shift(us[idx]), ul_lists[idx].pop()]),
                                                           conv=nn.down_right_shifted_conv2d)
                            uls[idx] = extra_nin(uls[idx], extra)
                        else:
                            us[idx] = nn.aux_gated_resnet(
                                us[idx],
                                tf.concat(3, [u_lists[idx].pop(), us[idx - 1]]),
                                conv=nn.down_shifted_conv2d
                            )
                            us[idx] = extra_nin(us[idx], extra)
                            uls[idx] = nn.aux_gated_resnet(uls[idx],
                                                           tf.concat(3, [nn.down_shift(us[idx]), ul_lists[idx].pop()]),
                                                           conv=nn.down_right_shifted_conv2d)
                            uls[idx] = extra_nin(uls[idx], extra)

                for u_list in u_lists.values():
                    assert len(u_list) == 0
                for ul_list in ul_lists.values():
                    assert len(ul_list) == 0

            for idx in range(1, len(self.nr_extra_nins)):
                uls[idx] = nn.aux_gated_resnet(uls[idx], uls[idx - 1], conv=nn.nin)

            x_out = nn.nin(nn.concat_elu(uls[-1]), self.nr_filters)

        return x_out

    def cond(self, x, c):
        import sandbox.pchen.InfoGAN.infogan.misc.imported.scopes as scopes
        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
        x = tf.reshape(
            x,
            [-1, ] + list(self._shape[:2]) + [self.nr_filters]
        )
        c = tf.reshape(
            c,
            [-1, ] + list(self._shape[:2]) + [self.nr_filters]
        )

        # old cond arch
        # counters = {}
        # with scopes.arg_scope(
        #         [nn.nin],
        #         counters=counters, init=self._custom_phase == CustomPhase.init, ema=None
        # ):
        #     gated = nn.aux_gated_resnet(x, c, conv=nn.nin)
        #     x_out = nn.nin(nn.concat_elu(gated), 10*self.nr_logistic_mix)

        # new cond arch
        counters = {}
        with scopes.arg_scope(
                [nn.nin],
                counters=counters, init=self._custom_phase == CustomPhase.init, ema=None
        ):
            gated = tf.concat(
                3,
                [
                    nn.aux_gated_resnet(x, c, conv=nn.nin),
                    nn.aux_gated_resnet(c, x, conv=nn.nin),
                ],
            )
            for _ in range(self.nr_cond_nins):
                gated = nn.gated_resnet(gated, conv=nn.nin)
            x_out = nn.nin(nn.concat_elu(gated), 10 * self.nr_logistic_mix)

        return x_out

    def logli(self, x_var, info):
        x_var = tf.reshape(
            x_var,
            [-1, ] + list(self._shape)
        ) * 2

        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn

        causal, cond = info["causal_feats"], info["cond_feats"]
        tgt_vec = self.cond_temp(causal, cond)
        logli = nn.discretized_mix_logistic(
            x_var,
            tgt_vec
        )
        return tf.reduce_sum(
            tf.reshape(logli, [-1, self._shape[0] * self._shape[1]]),
            reduction_indices=1
        )

    @functools.lru_cache(maxsize=None)
    def sample_sym(self, n, unconditional=False, deep_cond=False):
        x_var, context_var = \
            tf.placeholder(tf.float32, shape=[n, ] + list(self._shape)), \
            tf.placeholder(tf.float32, shape=[n, ] + list(self._shape[:2]) + [self.nr_filters])
        causal = self.infer_temp(x_var, context=context_var if deep_cond else None)
        if unconditional:
            context_var = context_var * 0.
        tgt_vec = self.cond_temp(causal, context_var)
        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
        return x_var, \
               context_var, \
               nn.sample_from_discretized_mix_logistic(tgt_vec, self.nr_logistic_mix) / 2, \
               tgt_vec

    @functools.lru_cache(maxsize=None)
    def sample_one_step(self, x_var, info):
        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn

        assert "causal_feats" not in info
        cond_feats = info["cond_feats"]
        causal_feats = self.infer_temp(x_var)
        tgt_vec = self.cond_temp(causal_feats, cond_feats)
        return nn.sample_from_discretized_mix_logistic(tgt_vec, self.nr_logistic_mix) / 2  # convert back to 0.5 scale

    def prior_dist_info(self, batch_size):
        return {}

    def sample_logli(self, info):
        raise NotImplemented

    @property
    def dist_info_keys(self):
        return ["context"] if self._context else []

    @property
    def dist_flat_dim(self):
        return self._shape[0] * self._shape[1] * self._context_dim

    def activate_dist(self, flat):
        return dict(context=flat)

class GeneralizedShearingFlow(Distribution):
    def __init__(
            self,
            base_dist,
            nn_builder,
            condition_fn,
            effect_fn,
            combine_fn,
            backwad_join_fn,
            forward_join_fn,
    ):
        # fix me later about how to serailzie fn?
        # Serializable.quick_init(self, locals())

        global G_IDX
        G_IDX += 1
        self._name = "Shearing_%s" % (G_IDX)
        assert base_dist
        assert nn_builder
        assert condition_fn
        assert effect_fn
        assert combine_fn
        self._base_dist = base_dist
        self._nn_template = tf.make_template(self._name, nn_builder)
        self._condition_set = condition_fn
        self._effect_set = effect_fn
        self._combine = combine_fn
        self._backward_join = backwad_join_fn
        self._forward_join = forward_join_fn

        self.train_mode()

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init
        self._base_dist.init_mode()

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train
        self._base_dist.train_mode()

    @property
    def dim(self):
        return self._base_dist.dim

    @property
    def effective_dim(self):
        return self._base_dist.effective_dim

    def infer(self, x_var, dist_info):
        condition = self._condition_set(x_var)
        effect = self._effect_set(x_var)

        counters = {}
        with scopes.default_arg_scope(
                counters=counters, init=self._custom_phase == CustomPhase.init,
                ema=None
        ):
            join_params = self._nn_template(condition)

        return condition, effect, join_params

    def logli(self, x_var, dist_info):
        condition, effect, join_params = self.infer(x_var, dist_info)
        effect_shp = nn.int_shape(effect)
        joined, logdiff = self._backward_join(effect, join_params)
        eps = self._combine(condition, joined)

        return self._base_dist.logli(eps, dist_info) + logdiff

    def logli_init_prior(self, x_var):
        return self._base_dist.logli_init_prior(x_var)

    def prior_dist_info(self, batch_size):
        return self._base_dist.prior_dist_info(batch_size)

    def sample_logli(self, dist_info):
        eps, logpeps = self._base_dist.sample_logli(dist_info)
        condition, effect, join_params = self.infer(eps, dist_info)
        effect_shp = nn.int_shape(effect)
        joined, logdiff = self._forward_join(effect, join_params)
        x = self._combine(condition, joined)

        return x, logpeps + logdiff

    @property
    def dist_info_keys(self):
        return self._base_dist.dist_info_keys

    @property
    def dist_flat_dim(self):
        return self._base_dist.dist_flat_dim

    def activate_dist(self, flat):
        return self._base_dist.activate_dist(flat)

    def nonreparam_logli(self, x_var, dist_info):
        raise "not defined"

def LinearShearingFlow(
        base_dist,
        nn_builder,
        condition_fn,
        effect_fn,
        combine_fn,
):
    def backwad_join_fn(effect, join_params):
        chns = int_shape(join_params)[-1] // 2
        mu = join_params[:, :, :, chns:]
        logstd = tf.tanh(join_params[:, :, :, :chns])
        joined = effect * tf.exp(logstd) + mu
        return joined, tf.reduce_sum(logstd, [1,2,3])

    def forward_join_fn(effect, join_params):
        chns = int_shape(join_params)[-1] // 2
        mu = join_params[:, :, :, chns:]
        logstd = tf.tanh(join_params[:, :, :, :chns])
        joined = (effect - mu) / tf.exp(logstd)
        return joined, tf.reduce_sum(logstd, [1,2,3])

    return GeneralizedShearingFlow(
        base_dist,
        nn_builder,
        condition_fn,
        effect_fn,
        combine_fn,
        backwad_join_fn=backwad_join_fn,
        forward_join_fn=forward_join_fn,
    )

def LeakyLinearShearingFlow(
        base_dist,
        nn_builder,
        condition_fn,
        effect_fn,
        combine_fn,
        leakiness=1.,
):
    def backwad_join_fn(effect, join_params):
        chns = int_shape(join_params)[-1] // 2
        mu = join_params[:, :, :, chns:]
        logstd = tf.tanh(join_params[:, :, :, :chns])
        logstd = tf.select(
            effect >= 0,
            logstd,
            tf.ones_like(logstd) * tf.log(leakiness)
        )
        joined = effect * tf.exp(logstd) + mu
        return joined, tf.reduce_sum(logstd, [1,2,3])

    def forward_join_fn(effect, join_params):
        chns = int_shape(join_params)[-1] // 2
        mu = join_params[:, :, :, chns:]
        logstd = tf.tanh(join_params[:, :, :, :chns])
        logstd = tf.select(
            (effect - mu) >= 0,
            logstd,
            tf.ones_like(logstd) * tf.log(leakiness)
        )
        joined = (effect - mu) / tf.exp(logstd)
        return joined, tf.reduce_sum(logstd, [1,2,3])

    return GeneralizedShearingFlow(
        base_dist,
        nn_builder,
        condition_fn,
        effect_fn,
        combine_fn,
        backwad_join_fn=backwad_join_fn,
        forward_join_fn=forward_join_fn,
    )

class ShearingFlow(Distribution):
    def __init__(
            self,
            base_dist,
            nn_builder,
            condition_fn,
            effect_fn,
            combine_fn,
    ):
        # fix me later about how to serailzie fn?
        # Serializable.quick_init(self, locals())

        global G_IDX
        G_IDX += 1
        self._name = "Shearing_%s" % (G_IDX)
        assert base_dist
        assert nn_builder
        assert condition_fn
        assert effect_fn
        assert combine_fn
        self._base_dist = base_dist
        self._nn_template = tf.make_template(self._name, nn_builder)
        self._condition_set = condition_fn
        self._effect_set = effect_fn
        self._combine = combine_fn

        self.train_mode()

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init
        self._base_dist.init_mode()

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train
        self._base_dist.train_mode()

    @property
    def dim(self):
        return self._base_dist.dim

    @property
    def effective_dim(self):
        return self._base_dist.effective_dim

    def infer(self, x_var, dist_info):
        condition = self._condition_set(x_var)
        effect = self._effect_set(x_var)

        counters = {}
        with scopes.default_arg_scope(
                counters=counters, init=self._custom_phase == CustomPhase.init,
                ema=None
        ):
            mu, logstd = self._nn_template(condition)

        return condition, effect, mu, logstd

    def logli(self, x_var, dist_info):
        condition, effect, mu, logstd = self.infer(x_var, dist_info)
        effect_shp = nn.int_shape(effect)
        eps = self._combine(condition, effect * tf.exp(logstd) + mu)

        return self._base_dist.logli(eps, dist_info) + \
               tf.reduce_sum(
                   tf.reshape(logstd, [-1, np.prod(effect_shp[1:])]),
                   reduction_indices=1
               )

    def logli_init_prior(self, x_var):
        return self._base_dist.logli_init_prior(x_var)

    def prior_dist_info(self, batch_size):
        return self._base_dist.prior_dist_info(batch_size)

    def sample_logli(self, dist_info):
        eps, logpeps = self._base_dist.sample_logli(dist_info)
        condition, effect, mu, logstd = self.infer(eps, dist_info)
        effect_shp = nn.int_shape(effect)
        x = self._combine(condition, (effect - mu) / tf.exp(logstd))

        return x, logpeps + \
               tf.reduce_sum(
                   tf.reshape(logstd, [-1, np.prod(effect_shp[1:])]),
                   reduction_indices=1
               )

    @property
    def dist_info_keys(self):
        return self._base_dist.dist_info_keys

    @property
    def dist_flat_dim(self):
        return self._base_dist.dist_flat_dim

    def activate_dist(self, flat):
        return self._base_dist.activate_dist(flat)

    def nonreparam_logli(self, x_var, dist_info):
        raise "not defined"


class ReshapeFlow(Distribution):
    def __init__(
            self,
            base_dist,
            forward_fn,
            backward_fn,
            debug=False,
            name="reshape",
    ):
        global G_IDX
        G_IDX += 1
        self._name = "%s_%s" % (name, G_IDX)
        self._base_dist = base_dist
        def _this_template(mode, *args, **kwargs):
            if mode == "forward":
                return forward_fn(*args, **kwargs)
            elif mode == "backward":
                return backward_fn(*args, **kwargs)

            assert False
        this_template = tf.make_template(self._name, _this_template)

        self._forward = functools.partial(this_template, "forward")
        self._backward = functools.partial(this_template, "backward")
        self._debug = debug

        self.train_mode()

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init
        self._base_dist.init_mode()

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train
        self._base_dist.train_mode()

    @property
    def dim(self):
        return self._base_dist.dim

    @property
    def effective_dim(self):
        return self._base_dist.effective_dim

    def logli(self, x_var, dist_info):
        with scopes.default_arg_scope(
                counters={}, init=self._custom_phase == CustomPhase.init,
                ema=None
        ):
            arr_maybe = self._backward(x_var)
            if isinstance(arr_maybe, (tuple, list)):
                eps, log_diff = arr_maybe
            else:
                eps = arr_maybe
                log_diff = 0.
            log_diff = tf.Print(log_diff, [x_var, log_diff]) if self._debug else log_diff
            return self._base_dist.logli(eps, dist_info) + log_diff

    def sample_logli(self, dist_info):
        with scopes.default_arg_scope(
                counters={}, init=self._custom_phase == CustomPhase.init,
                ema=None
        ):
            eps, logpeps = self._base_dist.sample_logli(dist_info)
            arr_maybe = self._forward(eps)
            if isinstance(arr_maybe, (tuple, list)):
                x, log_diff = arr_maybe
            else:
                x = arr_maybe
                log_diff = 0.
            return x, logpeps + log_diff

    def prior_dist_info(self, batch_size):
        return self._base_dist.prior_dist_info(batch_size)

    @property
    def dist_info_keys(self):
        return self._base_dist.dist_info_keys

    @property
    def dist_flat_dim(self):
        return self._base_dist.dist_flat_dim

    def activate_dist(self, flat):
        return self._base_dist.activate_dist(flat)

    def nonreparam_logli(self, x_var, dist_info):
        raise "not defined"


class OldDequantizedFlow(Distribution):
    def __init__(
            self,
            base_dist,
            width=1. / 256,
    ):
        global G_IDX
        G_IDX += 1
        self._name = "OldDequantized_%s" % (G_IDX)
        self._base_dist = base_dist
        self._width = width

    @overrides
    def init_mode(self):
        self._base_dist.init_mode()

    @overrides
    def train_mode(self):
        self._base_dist.train_mode()

    @property
    def dim(self):
        return self._base_dist.dim

    @property
    def effective_dim(self):
        return self._base_dist.effective_dim

    def logli(self, x_var, dist_info):
        return self._base_dist.logli(x_var, dist_info) + self.dim * np.log(self._width)

    def sample_logli(self, dist_info):
        x, logpeps = self._base_dist.sample_logli(dist_info)
        return x, logpeps + self.dim * np.log(self._width)

    def prior_dist_info(self, batch_size):
        return self._base_dist.prior_dist_info(batch_size)

    @property
    def dist_info_keys(self):
        return self._base_dist.dist_info_keys

    @property
    def dist_flat_dim(self):
        return self._base_dist.dist_flat_dim

    def activate_dist(self, flat):
        return self._base_dist.activate_dist(flat)

    def nonreparam_logli(self, x_var, dist_info):
        raise "not defined"

class DequantizedFlow(Distribution):
    def __init__(
            self,
            base_dist,
            noise_dist,
    ):
        global G_IDX
        G_IDX += 1
        self._name = "Dequantized_%s" % (G_IDX)
        self._base_dist = base_dist
        self._noise_dist = noise_dist

    @overrides
    def init_mode(self):
        self._base_dist.init_mode()
        self._noise_dist.init_mode()

    @overrides
    def train_mode(self):
        self._base_dist.train_mode()
        self._noise_dist.train_mode()

    @property
    def dim(self):
        return self._base_dist.dim

    @property
    def effective_dim(self):
        return self._base_dist.effective_dim

    def logli(self, x_var, dist_info):
        # abusing the notation in the sense that this is a vlb
        eps, eps_logli = self._noise_dist.sample_logli(dict(condition=x_var))
        return self._base_dist.logli(x_var+eps, dist_info) - eps_logli

    def sample_logli(self, dist_info):
        # abusing the notation in the sense that it doesn't
        #  quantize the output again
        x, logpeps = self._base_dist.sample_logli(dist_info)
        return x, logpeps

    def prior_dist_info(self, batch_size):
        return self._base_dist.prior_dist_info(batch_size)

    @property
    def dist_info_keys(self):
        return self._base_dist.dist_info_keys

    @property
    def dist_flat_dim(self):
        return self._base_dist.dist_flat_dim

    def activate_dist(self, flat):
        return self._base_dist.activate_dist(flat)

    def nonreparam_logli(self, x_var, dist_info):
        raise "not defined"

class DequantizationDistribution(Distribution):
    @property
    def dist_info_keys(self):
        return ["condition"]

class UniformDequant(DequantizationDistribution):
    def __init__(
            self,
            width=1. / 256,
    ):
        global G_IDX
        G_IDX += 1
        self._name = "UniformDequant_%s" % (G_IDX)
        self._width = width

    def sample_logli(self, dist_info):
        condition = dist_info["condition"]
        shp = int_shape(condition)
        dim = np.prod(shp[1:])
        return tf.random_uniform(shp, maxval=self._width), (-dim * np.log(self._width)).astype(np.float32)

class FixedSpatialTruncatedLogisticDequant(DequantizationDistribution):
    def __init__(
            self,
            shape,
            width=1. / 256,
            scale=1.,
    ):
        global G_IDX
        G_IDX += 1
        self._name = "TruncatedLogisticDequant_%s" % (G_IDX)
        self._width = width
        dim = np.prod(shape)
        self._shape = shape
        self._dim = dim
        self._scale = scale
        pshp = [1, dim]
        self._mu = tf.get_variable(
            self._name + "_mu",
            pshp,
            tf.float32,
            tf.random_normal_initializer(0, 0.05),
            trainable=True
        )
        self._log_scale = tf.get_variable(
            self._name + "_log_scale",
            pshp,
            tf.float32,
            tf.random_normal_initializer(0, 0.05),
            trainable=True
        )
        self._delegate_dist = TruncatedLogistic(shape, -1., 1.)
        self.init_mode()

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init
        self._delegate_dist.init_mode()

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train
        self._delegate_dist.train_mode()

    def prior_dist_info(self, batch_size):
        def expand(x):
            if self._custom_phase == CustomPhase.init:
                x = x.initialized_value()
            return tf.tile(x, [batch_size, 1])
        return dict(
            mu=expand(self._mu),
            scale=tf.exp(expand(self._log_scale)) * self._scale,
        )

    def sample_logli(self, dist_info):
        condition = dist_info["condition"]
        delegate_info = self.prior_dist_info(int_shape(condition)[0])
        eps, logli_eps = self._delegate_dist.sample_logli(
            delegate_info
        )
        scaling = self._width / 2.
        return (eps+1.) * scaling, logli_eps - tf.log(scaling) * self._dim

class FactorizedEncodingSpatialPiecewiseLinearDequant(DequantizationDistribution):
    def __init__(
            self,
            shape,
            nn_builder,
            width=1. / 256,
            nr_mixtures=1,
    ):
        global G_IDX
        G_IDX += 1
        self._name = "FactorizedEncodingPiecewiseLinearDequant_%s" % (G_IDX)
        self._nn_template = tf.make_template(self._name, nn_builder)
        self._width = width
        dim = np.prod(shape)
        self._shape = shape
        self._dim = dim
        self._nr_mixtures = nr_mixtures

        assert nr_mixtures == 1
        self._delegate_dist = PiecewiseLinear(shape, )
        # if nr_mixtures == 1:
        #     self._delegate_dist = truncatedlogistic(shape, -1., 1.)
        # else:
        # self._delegate_dist = Mixture(
        #     [
        #         (
        #             TruncatedLogistic(
        #                 shape, -1., 1.,
        #             ),
        #             1./nr_mixtures
        #         )
        #         for _ in range(nr_mixtures)
        #         ]
        # )

        self.init_mode()

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init
        self._delegate_dist.init_mode()

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train
        self._delegate_dist.train_mode()

    def sample_logli(self, dist_info):
        condition = dist_info["condition"]
        with scopes.default_arg_scope(
                counters={}, init=self._custom_phase == CustomPhase.init,
                ema=None
        ):
            info_tensor = self._nn_template(condition)
            if self._nr_mixtures != 1:
                # ensure the mixture reshape is correct
                shp = int_shape(info_tensor)
                ndim = len(shp)
                assert shp[-1] == self._nr_mixtures
                info_tensor = tf.transpose(
                    info_tensor,
                    [0, ndim-1, ] + list(range(1, ndim-1))
                )
            delegate_info_flat = tf.reshape(
                info_tensor,
                [-1, self._delegate_dist.dist_flat_dim]
            )
        eps, logli_eps = self._delegate_dist.sample_logli(
            self._delegate_dist.activate_dist(delegate_info_flat)
        )
        scaling = self._width / 2.
        return (eps+1.) * scaling, logli_eps - tf.log(scaling) * self._dim

class FactorizedEncodingSpatialKumaraswamyDequant(DequantizationDistribution):
    def __init__(
            self,
            shape,
            nn_builder,
            width=1. / 256,
            nr_mixtures=1,
    ):
        global G_IDX
        G_IDX += 1
        self._name = "FactorizedEncodingKumarasawamyDequant_%s" % (G_IDX)
        self._nn_template = tf.make_template(self._name, nn_builder)
        self._width = width
        dim = np.prod(shape)
        self._shape = shape
        self._dim = dim
        self._nr_mixtures = nr_mixtures

        # assert nr_mixtures == 1
        # self._delegate_dist = Kumaraswamy(shape, )
        self._delegate_dist = Mixture(
            [
                (
                    Kumaraswamy(
                        shape,
                    ),
                    1./nr_mixtures
                )
                for _ in range(nr_mixtures)
                ]
        )
        # if nr_mixtures == 1:
        #     self._delegate_dist = truncatedlogistic(shape, -1., 1.)
        # else:
        # self._delegate_dist = Mixture(
        #     [
        #         (
        #             TruncatedLogistic(
        #                 shape, -1., 1.,
        #             ),
        #             1./nr_mixtures
        #         )
        #         for _ in range(nr_mixtures)
        #         ]
        # )

        self.init_mode()

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init
        self._delegate_dist.init_mode()

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train
        self._delegate_dist.train_mode()

    def sample_logli(self, dist_info):
        condition = dist_info["condition"]
        with scopes.default_arg_scope(
                counters={}, init=self._custom_phase == CustomPhase.init,
                ema=None
        ):
            info_tensor = self._nn_template(condition)
            if self._nr_mixtures != 1:
                # ensure the mixture reshape is correct
                shp = int_shape(info_tensor)
                ndim = len(shp)
                assert shp[-1] == self._nr_mixtures
                info_tensor = tf.transpose(
                    info_tensor,
                    [0, ndim-1, ] + list(range(1, ndim-1))
                )
            delegate_info_flat = tf.reshape(
                info_tensor,
                [-1, self._delegate_dist.dist_flat_dim]
            )
        eps, logli_eps = self._delegate_dist.sample_logli(
            self._delegate_dist.activate_dist(delegate_info_flat)
        )
        scaling = self._width
        return (eps) * scaling, logli_eps - tf.log(scaling) * self._dim

class FactorizedEncodingSpatialTruncatedLogisticDequant(DequantizationDistribution):
    def __init__(
            self,
            shape,
            nn_builder,
            width=1. / 256,
            nr_mixtures=1,
    ):
        global G_IDX
        G_IDX += 1
        self._name = "FactorizedEncodingTruncatedLogisticDequant_%s" % (G_IDX)
        self._nn_template = tf.make_template(self._name, nn_builder)
        self._width = width
        dim = np.prod(shape)
        self._shape = shape
        self._dim = dim
        self._nr_mixtures = nr_mixtures

        # self._delegate_dist = TruncatedLogistic(shape, -1., 1.)
        # if nr_mixtures == 1:
        #     self._delegate_dist = truncatedlogistic(shape, -1., 1.)
        # else:
        self._delegate_dist = Mixture(
           [
               (
                   TruncatedLogistic(
                       shape, -1., 1.,
                   ),
                   1./nr_mixtures
               )
               for _ in range(nr_mixtures)
           ]
        )

        self.init_mode()

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init
        self._delegate_dist.init_mode()

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train
        self._delegate_dist.train_mode()

    def sample_logli(self, dist_info):
        condition = dist_info["condition"]
        with scopes.default_arg_scope(
                counters={}, init=self._custom_phase == CustomPhase.init,
                ema=None
        ):
            info_tensor = self._nn_template(condition)
            if self._nr_mixtures != 1:
                # ensure the mixture reshape is correct
                shp = int_shape(info_tensor)
                ndim = len(shp)
                assert shp[-1] == self._nr_mixtures
                info_tensor = tf.transpose(
                    info_tensor,
                    [0, ndim-1, ] + list(range(1, ndim-1))
                )
            delegate_info_flat = tf.reshape(
                info_tensor,
                [-1, self._delegate_dist.dist_flat_dim]
            )
        eps, logli_eps = self._delegate_dist.sample_logli(
            self._delegate_dist.activate_dist(delegate_info_flat)
        )
        scaling = self._width / 2.
        return (eps+1.) * scaling, logli_eps - tf.log(scaling) * self._dim


class FlowBasedDequant(DequantizationDistribution):
    def __init__(
            self,
            shape,
            context_processor,
            flow_builder,
            width=1. / 256,

    ):
        global G_IDX
        G_IDX += 1
        self._name = "FlowBasedDequant_%s" % (G_IDX)
        self._width = width
        dim = np.prod(shape)

        self._shape = shape
        self._dim = dim
        self._context_processor = tf.make_template(self._name + "_cp", context_processor)

        self._delegate = flow_builder()

        self.init_mode()

    @overrides
    def init_mode(self):
        self._custom_phase = CustomPhase.init
        self._delegate.init_mode()

    @overrides
    def train_mode(self):
        self._custom_phase = CustomPhase.train
        self._delegate.train_mode()

    def sample_logli(self, dist_info):
        condition = dist_info["condition"]
        bs = universal_int_shape(condition)[0]
        condition = tf.reshape(
            condition, [bs, 32, 32, 3]
        )

        with scopes.default_arg_scope(
                counters={}, init=self._custom_phase == CustomPhase.init
        ):
            processed = self._context_processor(condition)

        with scopes.default_arg_scope(
            context=processed,
        ):
            return self._delegate.sample_logli(
                self._delegate.prior_dist_info(batch_size=bs)
            )

# TODO: this has wrong impl for sampling
def normalize_legacy(dist):
    def normalize_per_dim(x):
        mu, inv_std = nn.init_normalization(x)
        return -mu, tf.log(inv_std)
    return ShearingFlow(
        dist,
        nn_builder=normalize_per_dim,
        condition_fn=lambda x: x,
        effect_fn=lambda x: x,
        combine_fn=lambda _, x: x,
    )

def normalize(dist):
    def normalize_per_dim(x):
        mu, inv_std = nn.init_normalization(x)
        return mu, inv_std

    init_mode = []
    @scopes.add_arg_scope_only("init")
    def forward(eps, init, ):
        if init:
            logger.log("Init used in sampling pass")
            assert len(init_mode) == 0
            init_mode.append("forward")
        else:
            assert len(init_mode) == 1
        mu, inv_std = normalize_per_dim(eps)
        sum_log_inv_std = tf.reduce_sum(tf.log(inv_std))
        if init_mode[0] == "forward":
            return (eps - mu) * inv_std, -sum_log_inv_std
        else:
            return eps / inv_std + mu, sum_log_inv_std

    @scopes.add_arg_scope_only("init")
    def backward(x, init, ):
        if init:
            logger.log("Init used in inference pass")
            assert len(init_mode) == 0
            init_mode.append("backward")
        else:
            assert len(init_mode) == 1
        mu, inv_std = normalize_per_dim(x)
        sum_log_inv_std = tf.reduce_sum(tf.log(inv_std))
        if init_mode[0] == "backward":
            return (x - mu) * inv_std, sum_log_inv_std
        else:
            return x / inv_std + mu, -sum_log_inv_std

    return ReshapeFlow(
        dist,
        forward_fn=forward,
        backward_fn=backward,
        name="normalize",
    )


def shift(dist, offset=0.5 + (1/256/2)):
    return ReshapeFlow(
        dist,
        forward_fn=lambda eps: eps - offset,
        backward_fn=lambda x: x + offset,
        debug=False,
        name="shift",
    )

def logitize(dist, coeff=0.90):
    # apply logit(coeff*x)
    # logli_diff_fn = lambda x: tf.reduce_sum(
    #     -tf.log(x - coeff*(x**2)),
    #     reduction_indices=[1,2,3]
    # )
    logli_diff_fn = lambda x: tf.reduce_sum(
        -safe_log(x*coeff) - safe_log(1-x*coeff) \
          + tf.log(coeff),
        reduction_indices=[1,2,3]
    )
    def forward(eps):
        x = tf.nn.sigmoid(-eps) / coeff
        return x, logli_diff_fn(x)
    return ReshapeFlow(
        dist,
        forward_fn=forward,
        backward_fn=lambda x: (tf.log(coeff*x) - tf.log(1-coeff*x), logli_diff_fn(x)),
        debug=False,
        name="logit",
    )

