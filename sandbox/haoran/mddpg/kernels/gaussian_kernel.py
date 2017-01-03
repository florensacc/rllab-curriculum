from sandbox.haoran.mddpg.kernels.base import Kernel
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np

class GaussianKernel(Kernel):
    """ a.k.a RBF kernel """
    def __init__(self, scope_name, L):
        Serializable.quick_init(self, locals())
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.L = tf.Variable(L, name='L')
        self.dim = L.shape[0]

    def get_kappa(self,xs):
        """
        N: sample
        K: head
        d: acd_dim
        Input xs: (N,K,d)
        Output: (N, j:other_dim, k:cur_dim)
        """
        L_expanded = tf.expand_dims(self.L, dim=0)
        xs_weighted = tf.batch_matmul(xs, L_expanded) # N x K X d
        # transforms xj to yj = L^T xj
        r = tf.reduce_sum(
            tf.square(xs_weighted),
            2, keep_dims=True
        ) # N x K x 1, computes yj^2
        D = -0.5 * (
            r - # yj^2
            2 * tf.batch_matmul(xs_weighted, xs_weighted, adj_y=True) + # yj*yk
            tf.transpose(r,[0,2,1]) # yk^2
        )
        kappa = tf.exp(D)
        return kappa

    def get_kappa_grads(self,xs):
        """
        Input xs: (N,K,d)
        Output: (N,K(j),K(k),d): \nabla_{xj}\kappa(xj,xk)
        """
        raise NotImplementedError


class DiagonalGaussianKernel(Kernel):
    """
    We use a separate class from generic GaussianKernel, as tensorflow doesn't
    know to perform sparse matrix multiply well (reported by Tianhao).
    """
    def __init__(self, scope_name, diag):
        Serializable.quick_init(self, locals())
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.diag = tf.Variable(
                diag,
                name='diag',
                dtype=tf.float32
            )
            self.diag_kappa = tf.expand_dims(
                tf.expand_dims(self.diag, 0),0) # 1 x 1 x d
            self.diag_kappa_grads = self.diag
            for i in range(3):
                self.diag_kappa_grads= tf.expand_dims(self.diag_kappa_grads, 0)
                    # 1 x 1 x 1 x d

    def get_kappa(self,xs):
        """
        N: sample
        K: head
        d: acd_dim
        Input xs: (N,K,d)
        Output: (N, j:other_dim, k:cur_dim) (N x K x K)
        """
        xs_weighted = xs * tf.sqrt(self.diag_kappa) # N x K X d
        r = tf.reduce_sum(
            tf.square(xs_weighted),
            2, keep_dims=True
        ) # N x K x 1, computes yj^2
        D = -0.5 * (
            r - # yj^2
            2 * tf.batch_matmul(xs_weighted, xs_weighted, adj_y=True) + # yj*yk
            tf.transpose(r,[0,2,1]) # yk^2
        )
        kappa = tf.exp(D)
        return kappa

    def get_kappa_grads(self,xs):
        """
        Input xs: (N,K,d)
        Output: (N,K(j),K(k),d): \nabla_{xj}\kappa(xj,xk)
        """
        kappa = tf.expand_dims(self.get_kappa(xs),dim=3) # N x K x K x 1
        xs_other = tf.expand_dims(xs, dim=1) # N x 1 x K x d
        xs_cur = tf.expand_dims(xs, dim=2) # N x K x 1 x d

        # tf hasn't implemented (N,1,K,d) - (N,K,1,d); need to do it one dim
        # at a time
        K = int(xs.get_shape()[1])
        diff = tf.pack(
            [xs_cur[:,:,0,:] - xs_other[:,:,k,:] for k in range(K)],
            # [xs_other[:,:,k,:] - xs_cur[:,:,0,:] for k in range(K)],
            axis=2
        ) # N x K x K x d: (x_j - x_k)
        kappa_grads = - kappa *  self.diag_kappa_grads * diff
        return kappa_grads

from scipy.spatial import distance
class SimpleAdaptiveDiagonalGaussianKernel(DiagonalGaussianKernel):
    """
    Since tensorflow doesn't allow computing the median in the graph, we do it
    in np and then feed the values back
    """
    def __init__(self, scope_name, dim, h_min=1e-3):
        self.scope_name = scope_name
        self.dim = dim
        self.h_min = h_min

        Serializable.quick_init(self, locals())
        with tf.variable_scope(self.scope_name):
            self.diag_placeholder = tf.placeholder(tf.float32, (None, self.dim))
                # N x d
            self.diag_kappa = tf.expand_dims(
                self.diag_placeholder,
                dim=1,
            ) # N x 1 x d
            self.diag_kappa_grads = tf.expand_dims(
                self.diag_kappa,
                dim=1,
            ) # N x 1 x 1 x d

    def update(self, algo, actor_feed):
        xs = self.sess.run(algo.policy.output, actor_feed) # N x K x d
        N, K, d = xs.shape
        assert self.dim == d
        assert K > 1, "cannot compute pairwise distance if K = 1"

        # See SVGD for details:
        # basically, we want \sum_j \kappa(x_j, x_k) \approx 1
        # The current code may not be efficient enough. One possible fix
        # is to compute the pairwise distances over the entire batch.
        hs = []
        for x in xs:
            # x is K x D
            dist = distance.pdist(x)
            h = np.median(dist)**2 / np.log(K)
            h = max(h, self.h_min)
            hs.append(h)
        self.diags = np.outer(1./np.array(hs), np.ones(d))

        extra_feed = {
            self.diag_placeholder: self.diags
        }
        return extra_feed

class SimpleDiagonalConstructor(object):
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma

    def diag(self):
        diag = np.ones(self.dim) * (1./self.sigma ** 2)
        return diag

# test ----------------------------------------------------------------------
import unittest
class TestDiagonalGaussianKernel(unittest.TestCase):
    def test_computation(self):
        """
        Check whether the computation is correct
        """
        diag = np.array([0.1, 1]) ** 2
        kernel = DiagonalGaussianKernel('kernel',diag)
        xs = tf.placeholder(tf.float32, [None,2,2])
        kappa = kernel.get_kappa(xs)
        kappa_grads= kernel.get_kappa_grads(xs)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            feed = {xs: np.array([[[0,0],[1,-2]]])}
            kappa_val, kappa_grads_val = sess.run([kappa,kappa_grads], feed)

        kappa_expected = np.array([[[1.0, 0.1346602956955058],
                               [0.1346602956955058, 1.0]]],np.float32)

        kappa_grads_expected = np.array([[[[0, 0],
                                     [0.0013466, -0.26932059]],
                                    [[-0.0013466, 0.26932059],
                                     [0, 0]]]],np.float32)

        np.testing.assert_almost_equal(kappa_val, kappa_expected, decimal=6)
        np.testing.assert_almost_equal(
            kappa_grads_val, kappa_grads_expected, decimal=6)

    def test_computation_2(self):
        """
        Check whether the computation is correct
        """
        diag = np.array([1.]) ** 2
        kernel = DiagonalGaussianKernel('kernel',diag)
        xs = tf.placeholder(tf.float32, [None, 2, 1])
        kappa = kernel.get_kappa(xs)
        kappa_grads = kernel.get_kappa_grads(xs)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            feed = {xs: np.array([[[0],[1]]])}
            kappa_val, kappa_grads_val = sess.run([kappa,kappa_grads], feed)

        kappa_expected = np.array([[[1., np.exp(-0.5)],
                               [np.exp(-0.5), 1.0]]],np.float32)
        kappa_grads_expected = np.array([
            [
                [
                    [0],
                    [1 * np.exp(-0.5)]
                ],
                [
                    [- 1 * np.exp(-0.5)],
                    [0]
                ]
            ]
        ],np.float32)

        np.testing.assert_almost_equal(kappa_val, kappa_expected, decimal=6)
        np.testing.assert_almost_equal(
            kappa_grads_val, kappa_grads_expected, decimal=6)

        # diff_expected = np.array([
        #     [
        #         [[0.], [-1.]],
        #         [[1.], [0.]]
        #     ]
        # ])
        # np.testing.assert_almost_equal(
        #     diff_val, diff_expected, decimal=6
        # )


    def test_adaptive_kernel(self):
        """
        Test whether the adaptive kernel can choose different diags for
            different samples.
        """
        diags = np.array([[0.1, 1, 10], [1,1,1]])
        kernel_1 = DiagonalGaussianKernel('kernel_1',diags[0])
        kernel_2 = DiagonalGaussianKernel('kernel_2',diags[1])
        adaptive_kernel = SimpleAdaptiveDiagonalGaussianKernel('adaptive',3)
        xs = tf.placeholder(tf.float32, [None,2,3])

        x1 = [[0,0,0],[1,2,3]]
        x2 = [[1,0,1],[1,0,0]]
        feed1 = {xs: np.array([x1])}
        feed2 = {xs: np.array([x2])}
        feed_all = {
            xs: np.array([x1,x2]),
            adaptive_kernel.diag_placeholder: diags
        }

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            kappa_1_val, grads_1_val = sess.run(
                [kernel_1.get_kappa(xs),kernel_1.get_kappa_grads(xs)],
                feed1
            )
            kappa_2_val, grads_2_val = sess.run(
                [kernel_2.get_kappa(xs),kernel_2.get_kappa_grads(xs)],
                feed2
            )
            kappa_val, grads_val = sess.run(
                [adaptive_kernel.get_kappa(xs),
                 adaptive_kernel.get_kappa_grads(xs)],
                feed_all
            )

        np.testing.assert_almost_equal(
            np.concatenate([kappa_1_val, kappa_2_val], axis=0),
            kappa_val,
            decimal=6
        )
        np.testing.assert_almost_equal(
            np.concatenate([grads_1_val, grads_2_val], axis=0),
            grads_val,
            decimal=6,
        )

    def test_automatic_adaptive_kernel(self):
        """
        Test whether the adaptive kernel chooses diags automatically for
            different samples.
        """
        pass

if __name__ == '__main__':
    unittest.main()
