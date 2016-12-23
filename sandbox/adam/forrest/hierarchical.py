import theano.tensor as TT
import theano
import numpy as np
from rllab.misc import ext
from rllab.distributions.base import Distribution

TINY = np.float32(1e-8)

def forward_pass(message_prob, prev_alpha):
    temp = TT.sum(TT.tensordot(message_prob, prev_alpha, axes=[[0], [0]]), axis=2).T
    return temp / TT.sum(temp)

def backward_pass(message_prob, prev_beta):
    c = TT.tensordot(message_prob, prev_beta, axes=[[2, 1], [0, 1]])
    temp = TT.stacklists([c, c]).T
    return temp / TT.sum(temp)


def compute_gamma(alpha, beta):
    gamma = alpha * beta
    z = TT.sum(gamma) + TINY
    return gamma / z, z


def compute_xi(message_prob, alpha, beta, z):
    temp = TT.tensordot(alpha, beta, 0).dimshuffle(0, 2, 1, 3)
    return temp * message_prob.dimshuffle(0, 2, 'x', 1) / z


class Hierarchical(Distribution):
    def __init__(self, num_options, action_dim):
        self._num_options = num_options
        self._action_dim = action_dim
        # x = TT.dmatrix()
        # print_op = theano.printing.Print('W_O')
        # printed_x = print_op(x)
        # self._f = theano.function([x], printed_x)

    @property
    def dim(self):
        return self._action_dim

    # TODO Reimplement
    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_means_vec = [old_dist_info_vars["action_mean_%s" % o] for o in range(0, self._num_options)]
        old_log_stds_vec = [old_dist_info_vars["action_log_std_%s" % o] for o in range(0, self._num_options)]
        new_means_vec = [new_dist_info_vars["action_mean_%s" % o] for o in range(0, self._num_options)]
        new_log_stds_vec = [new_dist_info_vars["action_log_std_%s" % o] for o in range(0, self._num_options)]
        """
        Compute the KL divergence of two Gaussian mixture distributions
        """
        old_prob_var = old_dist_info_vars["markov_prob"]
        new_prob_var = new_dist_info_vars["markov_prob"]
        # Assume layout is N * A
        # D = TT.tensordot(old_prob_var, (TT.log(old_prob_var + TINY) - TT.log(new_prob_var + TINY)), axes=[[1], [1]])
        D = TT.sum(
            old_prob_var * (TT.log(old_prob_var + TINY) - TT.log(new_prob_var + TINY)),
            axis=-1
        )
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        for o in range(0, self._num_options):
            old_means = old_means_vec[o]
            old_log_stds = old_log_stds_vec[o]
            new_means = new_means_vec[o]
            new_log_stds = new_log_stds_vec[o]

            old_std = TT.exp(old_log_stds)
            new_std = TT.exp(new_log_stds)

            numerator = TT.square(old_means - new_means) + \
                        TT.square(old_std) - TT.square(new_std)
            denominator = 2 * TT.square(new_std) + 1e-8
            D += old_prob_var[:, o]*TT.sum(
                numerator / denominator + new_log_stds - old_log_stds, axis=-1)


            # numerator = TT.square(old_means[o] - new_means[o]) + \
            #             TT.square(TT.exp(old_log_stds[o])) - TT.square(TT.exp(new_log_stds[o]))
            # denominator = np.float32(2) * TT.square(TT.exp(new_log_stds[o])) + TINY
            # D += old_prob_var[:, o] * TT.sum(
            #     numerator / denominator + new_log_stds[o] - old_log_stds[o], axis=-1)
        return D

    def sampled_likelihood_sym(self, x_var, dist_info_vars):
        p = 0
        for o in range(self._num_options):
            action_means = dist_info_vars["action_mean_%s" % o]
            action_stds = TT.exp(dist_info_vars["action_log_std_%s" % o])
            zs = (x_var - action_means) / action_stds
            prob = dist_info_vars["markov_prob"][:, o]
            gaussian_prob = np.float32(1.0) / TT.sqrt(np.power(np.float32(2)*np.float32(np.pi), action_means.shape[-1]) *
                                          TT.square(TT.prod(action_stds, axis=-1))) * \
                TT.exp(-np.float32(0.5) * TT.sum(TT.square(zs), axis=-1))
            p += prob * gaussian_prob
        return p

    def sampled_likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        li_new = self.sampled_likelihood_sym(x_var, new_dist_info_vars)
        li_old = self.sampled_likelihood_sym(x_var, old_dist_info_vars)
        return li_new / (li_old + TINY)

    # def kl(self, old_dist_info, new_dist_info):
    #     old_means = old_dist_info["mean"]
    #     old_log_stds = old_dist_info["log_std"]
    #     new_means = new_dist_info["mean"]
    #     new_log_stds = new_dist_info["log_std"]
    #     """
    #     Compute the KL divergence of two multivariate Gaussian distribution with
    #     diagonal covariance matrices
    #     """
    #     old_std = np.exp(old_log_stds)
    #     new_std = np.exp(new_log_stds)
    #     # means: (N*A)
    #     # std: (N*A)
    #     # formula:
    #     # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
    #     # ln(\sigma_2/\sigma_1)
    #     numerator = np.square(old_means - new_means) + \
    #                 np.square(old_std) - np.square(new_std)
    #     denominator = 2 * np.square(new_std) + 1e-8
    #     return np.sum(
    #         numerator / denominator + new_log_stds - old_log_stds, axis=-1)
    #
    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return TT.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        message_prob = dist_info_vars['message_prob']

        a_result, f_updates = theano.scan(fn=forward_pass,
                                          outputs_info=message_prob[0, 0, :, :].T,
                                          sequences=message_prob[1:])
        alpha = TT.concatenate((message_prob[0, 0, :, :].dimshuffle('x', 1, 0), a_result))

        beta, b_updates = theano.scan(fn=backward_pass,
                                      outputs_info=TT.ones_like(message_prob[0, 0, :, :].T),
                                      sequences=message_prob[::-1])
        ([gamma, z], g_updates) = theano.scan(fn=compute_gamma, sequences=[alpha, beta[::-1]])

        xi_result, x_updates = theano.scan(fn=compute_xi, sequences=[message_prob[1:], alpha[:-1], beta[-2::-1], z[:-1]])
        xi = TT.concatenate((TT.zeros_like(xi_result[0]).dimshuffle('x', 0, 1, 2, 3), xi_result))

        q_a = np.float32(0)
        q_o = np.float32(0)
        q_b = np.float32(0)
        w_a = TT.sum(gamma, axis=2)
        print_op = theano.printing.Print('W_O')
        # w_o = print_op(gamma[:, :, 1])
        w_o = gamma[:, :, 1]

        q_o = TT.tensordot(w_o, TT.log(dist_info_vars['option_prob']+TINY), [[1], [1]])
        for o in range(0, self._num_options):
            means = dist_info_vars["action_mean_%s" % o]
            log_stds = dist_info_vars["action_log_std_%s" % o]
            zs = (x_var - means) / TT.exp(log_stds)
            q_a += w_a[:, o] * (- TT.sum(log_stds, axis=-1) -
                                np.float32(0.5) * TT.sum(TT.square(zs), axis=-1) -
                                np.float32(0.5) * means.shape[-1] * np.log(np.float32((2 * np.pi))))
            # option = print_op(TT.log(dist_info_vars['option_prob'] + TINY))
            # q_o += w_o[:, o] * option
            # q_o += w_o[:, o] * TT.log(dist_info_vars['option_prob'][:, o] + TINY)
            w_b = TT.sum(xi[:, o, :, :, :], axis=(1, 2))
            q_b += TT.tensordot(w_b, TT.log(dist_info_vars['termination_prob_%s' % o] + TINY), axes=[[1], [1]])
        return q_a + q_o + q_b

    # def sample(self, dist_info):
    #     means = dist_info["mean"]
    #     log_stds = dist_info["log_std"]
    #     rnd = np.random.normal(size=means.shape)
    #     return rnd * np.exp(log_stds) + means
    #
    # def log_likelihood(self, xs, dist_info):
    #     means = dist_info["mean"]
    #     log_stds = dist_info["log_std"]
    #     zs = (xs - means) / np.exp(log_stds)
    #     return - np.sum(log_stds, axis=-1) - \
    #            0.5 * np.sum(np.square(zs), axis=-1) - \
    #            0.5 * means.shape[-1] * np.log(2 * np.pi)
    #
    def entropy(self, dist_info):
        log_stds = dist_info["action_log_std_0"]
        return np.sum(log_stds + np.log(np.sqrt(np.float32(2 * np.pi * np.e))), axis=-1)
    #
    # def entropy_sym(self, dist_info_var):
    #     log_std_var = dist_info_var["log_std"]
    #     return TT.sum(log_std_var + TT.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    def conditional_entropy_sym(self, x_var, dist_info_vars):
        conditional_prob = []
        normalization = 0
        for o in range(self._num_options):
            action_means = dist_info_vars["action_mean_%s" % o]
            action_stds = TT.exp(dist_info_vars["action_log_std_%s" % o])
            zs = (x_var - action_means) / action_stds
            prob = dist_info_vars["markov_prob"][:, o]
            gaussian_prob = np.float32(1.0) / TT.sqrt(np.power(np.float32(2 * np.pi), action_means.shape[-1]) *
                                          TT.square(TT.prod(action_stds, axis=-1))) * \
                            TT.exp(-np.float32(0.5) * TT.sum(TT.square(zs), axis=-1))
            cond_prob = prob*gaussian_prob
            conditional_prob.append(cond_prob)
            normalization += cond_prob
        normalization += TINY
        conditional_entropy = np.float32(0)
        for o in range(self._num_options):
            conditional_entropy -= conditional_prob[o]/normalization*TT.log(conditional_prob[o]/normalization + TINY)
        return conditional_entropy

    def importance_entropy_sym(self, x_var, new_dist_info_vars, old_dist_info_vars):
        conditional_prob = []
        new_normalization = np.float32(0)
        old_normalization = np.float32(0)
        for o in range(self._num_options):
            old_action_means = old_dist_info_vars["action_mean_%s" % o]
            old_action_stds = TT.exp(old_dist_info_vars["action_log_std_%s" % o])
            old_zs = (x_var - old_action_means) / old_action_stds
            old_prob = old_dist_info_vars["markov_prob"][:, o]
            old_gaussian_prob = np.float32(1.0) / TT.sqrt(np.power(np.float32(2 * np.pi), old_action_means.shape[-1]) *
                                          TT.square(TT.prod(old_action_stds, axis=-1))) * \
                            TT.exp(-np.float32(0.5) * TT.sum(TT.square(old_zs), axis=-1))
            old_cond_prob = old_prob*old_gaussian_prob

            new_action_means = new_dist_info_vars["action_mean_%s" % o]
            new_action_stds = TT.exp(new_dist_info_vars["action_log_std_%s" % o])
            new_zs = (x_var - new_action_means) / new_action_stds
            new_prob = new_dist_info_vars["markov_prob"][:, o]
            new_gaussian_prob = np.float32(1.0) / TT.sqrt(np.power(np.float32(2 * np.pi), new_action_means.shape[-1]) *
                                          TT.square(TT.prod(new_action_stds, axis=-1))) * \
                            TT.exp(-np.float32(0.5) * TT.sum(TT.square(new_zs), axis=-1))
            new_cond_prob = new_prob*new_gaussian_prob
            conditional_prob.append(new_cond_prob)
            new_normalization += new_cond_prob
            old_normalization += old_cond_prob
        conditional_entropy = np.float32(0)
        new_normalization += TINY
        old_normalization += TINY
        for o in range(self._num_options):
            conditional_entropy -= conditional_prob[o]/old_normalization*TT.log(conditional_prob[o]/new_normalization + TINY)
        return conditional_entropy

    @property
    def dist_info_keys(self):
        return [("message_prob", 3), ("option_prob", 1), ("markov_prob", 1)] + \
            [("action_mean_%s" % o, 1) for o in range(0, self._num_options)] +  \
            [("action_log_std_%s" % o, 1) for o in range(0, self._num_options)] + \
            [("termination_prob_%s" % o, 1) for o in range(0, self._num_options)]
