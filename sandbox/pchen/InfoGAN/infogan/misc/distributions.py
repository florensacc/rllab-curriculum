from __future__ import print_function
from __future__ import absolute_import
import itertools
import tensorflow as tf
import numpy as np
import prettytensor as pt

TINY = 1e-8

floatX = np.float32


class Distribution(object):
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
        return self.kl(dist_info, self.prior_dist_info(dist_info.values()[0].get_shape()[0]))

    def logli(self, x_var, dist_info):
        """
        :param x_var:
        :param dist_info:
        :return: log likelihood of the data
        """
        raise NotImplementedError

    def logli_prior(self, x_var):
        return self.logli(x_var, self.prior_dist_info(x_var.get_shape()[0]))

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


class Categorical(Distribution):
    def __init__(self, dim):
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
        ids = tf.multinomial(tf.log(prob + TINY), num_samples=1)[:, 0]
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
            self._init_prior_mean = tf.reshape(tf.cast(prior_mean if init_prior_mean is None else init_prior_mean, floatX), [1, -1])
            self._init_prior_stddev = tf.reshape(tf.cast(prior_stddev if init_prior_stddev is None else init_prior_stddev, floatX), [1, -1])

            prior_mean = tf.get_variable(
                "prior_mean_%s" % self._name,
                initializer=tf.constant(prior_mean),
                dtype=floatX,
            )
            # forget it untill we code it numerically more stable param
            # prior_stddev = tf.get_variable("prior_stddev_%s" % self._name, initializer=prior_mean)
        self._prior_mean = tf.reshape(tf.cast(prior_mean, floatX), [1, -1])
        self._prior_stddev = tf.reshape(tf.cast(prior_stddev, floatX), [1, -1])

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

    def sample(self, dist_info):
        mean = dist_info["mean"]
        stddev = dist_info["stddev"]
        epsilon = tf.random_normal(tf.shape(mean))
        return mean + epsilon * stddev

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
        return tf.random_uniform([batch_size, self.dim], minval=-1., maxval=1.)


class Bernoulli(Distribution):
    def __init__(self, dim):
        self._dim = dim

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
        return tf.reduce_sum(
            x_var * tf.log(p + TINY) + (1.0 - x_var) * tf.log(1.0 - p + TINY),
            reduction_indices=1
        )

    def entropy(self, dist_info):
        prob = dist_info["p"]
        neg_prob = 1. - prob
        return -tf.reduce_sum(prob * tf.log(prob + TINY), reduction_indices=1) \
                -tf.reduce_sum(neg_prob * tf.log(neg_prob + TINY), reduction_indices=1)

    def nonreparam_logli(self, x_var, dist_info):
        return self.logli(x_var, dist_info)

    def activate_dist(self, flat_dist):
        return dict(p=tf.nn.sigmoid(flat_dist))

    def sample(self, dist_info):
        p = dist_info["p"]
        return tf.cast(tf.less(tf.random_uniform(p.get_shape()), p), tf.float32)

    def prior_dist_info(self, batch_size):
        return dict(p=0.5 * tf.ones([batch_size, self.dim]))

class MeanBernoulli(Bernoulli):
    """
    Behaves almost the same as the usual Bernoulli distribution, except that when sampling from it, directly
    return the mean instead of sampling binary values
    """

    def sample(self, dist_info):
        return dist_info["p"]

    def nonreparam_logli(self, x_var, dist_info):
        return tf.zeros_like(x_var[:, 0])


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
            for k, v in dist_info_i.iteritems():
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
    def __init__(self, pairs):
        assert len(pairs) >= 1
        self._pairs = pairs
        self._dim = pairs[0][0].dim
        self._dims = [dist.dist_flat_dim for dist, _ in pairs]
        self._dist_flat_dim = np.product(self._dims)
        for dist, p in pairs:
            assert self._dim == dist.dim

    def split(self, x):
        def go():
            i = 0
            for idim in self._dims:
                yield x[:, i:i+idim]
                i += idim
        return list(go())

    def merge(self, xs):
        return tf.concat(1, xs)

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self._dim

    def logli(self, x, dist_info):
        infos = dist_info["infos"]
        # li = 0.
        loglips = []
        for pair, idist_info in zip(self._pairs, infos):
            dist, p = pair
            # li += tf.exp(dist.logli(x, idist_info)) * p
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
        return dict(infos=[dist.prior_dist_info(batch_size) for dist, _ in self._pairs])

    def init_prior_dist_info(self, batch_size):
        return dict(infos=[dist.init_prior_dist_info(batch_size) for dist, _ in self._pairs])

    def nonreparam_logli(self, x_var, dist_info):
        return tf.zeros_like(x_var[:, 0])

    def kl(self, p, q):
        raise NotImplemented

    def sample(self, dist_info):
        infos = dist_info["infos"]
        samples = [
                pair[0].sample(idist) for pair, idist in zip(self._pairs, infos)
        ]
        bs = int(samples[0].get_shape()[0])
        prob = np.asarray([[p for _, p in self._pairs]]*bs)
        ids = tf.multinomial(tf.log(prob), num_samples=1)[:,0]
        onehot_table = tf.constant(np.eye(len(self._pairs), dtype=np.float32))
        onehot = tf.nn.embedding_lookup(onehot_table, ids)
        # return onehot, tf.constant(0.) + samples, tf.reduce_sum(
        #     tf.reshape(onehot, [bs, len(infos), 1]) * tf.transpose(samples, [1, 0, 2]),
        #     reduction_indices=1
        # )
        return tf.reduce_sum(
            tf.reshape(onehot, [bs, len(infos), 1]) * tf.transpose(samples, [1, 0, 2]),
            reduction_indices=1
        )

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
                yield flat_dist[:, i:i+dist.dist_flat_dim]
                i += dist.dist_flat_dim
        return dict(infos=list(go()))

class AR(Distribution):
    def __init__(
            self,
            dim,
            base_dist,
            depth=2,
            neuron_ratio=4,
            reverse=True,
    ):
        self._name = "%sD_AR_id_%s" % (dim, G_IDX)
        global G_IDX
        G_IDX += 1

        self._dim = dim
        self._base_dist = base_dist
        self._iaf_template = pt.template("y")
        self._depth = depth
        self._reverse = reverse

        assert depth >= 1
        with pt.defaults_scope(activation_fn=tf.nn.relu):
            for di in xrange(depth):
                self._iaf_template = \
                    self._iaf_template.arfc(
                        2*dim*neuron_ratio,
                        ngroups=dim,
                        zerodiagonal=di == 0, # only blocking the first layer can stop data flow
                    )
        self._iaf_template = \
            self._iaf_template.\
                arfc(
                    dim * 2,
                    activation_fn=None,
                    ngroups=dim,
                ).\
                reshape([-1, self._dim, 2]).\
                apply(tf.transpose, [0, 2, 1]).\
                reshape([-1, 2*self._dim])

    @property
    def dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self._dim

    def infer(self, x_var):
        flat_iaf = self._iaf_template.construct(y=x_var).tensor
        iaf_mu, iaf_logstd = flat_iaf[:, :self._dim], flat_iaf[:, self._dim:]
        return iaf_mu, (iaf_logstd)

    def logli(self, x_var, dist_info):
        iaf_mu, iaf_logstd = self.infer(x_var)
        z = x_var / tf.exp(iaf_logstd) - iaf_mu
        if self._reverse:
            z = tf.reverse(z, [False, True])
        return self._base_dist.logli_prior(z) - tf.reduce_sum(iaf_logstd, reduction_indices=1)

    def logli_init_prior(self, x_var):
        return self._base_dist.logli_init_prior(x_var)

    def prior_dist_info(self, batch_size):
        return dict(n=batch_size)

    def sample(self, info):
        return self.sample_n(info["n"])[0]


    def sample_n(self, n=100):
        try:
            z, logpz = self._base_dist.sample_n(n=n)
        except AttributeError:
            z = self._base_dist.sample_prior(batch_size=n)
            logpz = self._base_dist.logli_prior(z)
        if self._reverse:
            z = tf.reverse(z, [False, True])
        go = z # place holder
        for i in xrange(self._dim):
            # mask = np.zeros((n, self._dim))
            # mask[:, :(i+1)] = 1.
            iaf_mu, iaf_logstd = self.infer(go)
            # iaf_mu *= mask
            # iaf_std = mask * iaf_std
            go = iaf_mu + tf.exp(iaf_logstd)*z
        return go, logpz - tf.reduce_sum(iaf_logstd, reduction_indices=1)


    @property
    def dist_info_keys(self):
        return []

    def nonreparam_logli(self, x_var, dist_info):
        return tf.zeros_like(x_var[:, 0])

