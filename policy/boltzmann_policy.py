from .base import Policy
from misc.overrides import overrides
from misc.special import softmax, weighted_sample, cat_perplexity
from core.serializable import Serializable
import numpy as np
import tensorfuse as theano
import tensorfuse.tensor as T

class BoltzmannPolicy(Policy, Serializable):
    
    def __init__(self, qfunc, temperature=None):
        if temperature is None:
            temperature = 1e8
        temp_var = theano.shared(np.cast[theano.config.floatX](temperature), "temperature")
        prob_var = T.nnet.softmax(qfunc.qval_var / temp_var)

        self._compute_probs = theano.function([qfunc.input_var], prob_var, allow_input_downcast=True)
        self._qfunc = qfunc
        self._action_dim = qfunc.n_actions
        self._temp_var = temp_var
        self._pdist_var = prob_var
        Serializable.__init__(self, qfunc, temperature)

    # fit a temperature to the given Q-values so that the overall perplexity
    # is at most `threshold` less than the maximum perplexity (= action_dim)
    def learn_temperature(self, qval, threshold):
        # find an upper bound for the temperature
        def eval_perplexity(temp):
            return np.min(cat_perplexity(softmax(qval / temp)))
        perplexity_threshold = self._action_dim - threshold
        max_temp = 0.5
        max_perplexity = 0
        while max_perplexity < perplexity_threshold:
            max_temp = max_temp * 2
            max_perplexity = eval_perplexity(max_temp)
        # find a lower bound for the temperature
        min_temp = max_temp
        min_perplexity = max_perplexity
        while min_perplexity > perplexity_threshold:
            min_temp = min_temp / 2
            min_perplexity = eval_perplexity(min_temp)
        # do binary search to find a best temperature
        while max_temp - min_temp > 1e-3:
            mid_temp = 0.5 * (max_temp + min_temp)
            mid_perplexity = eval_perplexity(mid_temp)
            if mid_perplexity >= perplexity_threshold:
                max_temp = mid_temp
            else:
                min_temp = mid_temp
        theano.compat.set_value(self._temp_var, np.cast[theano.config.floatX](max_temp))

    @property
    def temperature(self):
        return theano.compat.get_value(self._temp_var)

    @temperature.setter
    def temperature(self, val):
        theano.compat.set_value(self._temp_var, np.cast[theano.config.floatX](val))

    @property
    def params(self):
        return self._qfunc.params + [self._temp_var]

    @property
    def self_params(self):
        return [self._temp_var]

    @property
    def action_dim(self):
        return self._action_dim

    @overrides
    def get_actions(self, observations):
        probs = self._compute_probs(observations)
        action_dim = self._n_actions
        actions = [weighted_sample(prob, range(action_dim)) for prob in probs]
        return actions, probs

    @overrides
    def kl(self, old_prob_var, new_prob_var):
        return T.sum(old_prob_var * (T.log(old_prob_var) - T.log(new_prob_var)), axis=1)

    @overrides
    def likelihood_ratio(self, old_prob_var, new_prob_var, action_var):
        N = old_prob_var.shape[0]
        return new_prob_var[T.arange(N), action_var.reshape((-1,))] / old_prob_var[T.arange(N), action_var.reshape((-1,))]

    @property
    def pdist_var(self):
        return self._pdist_var

    @overrides
    def get_param_values(self):
        return np.append(self._qfunc.get_param_values(), self.temperature)

    @overrides
    def set_param_values(self, flattened_params):
        self.temperature = flattened_params[-1]
        self._qfunc.set_param_values(flattened_params[:-1])

    def get_self_param_values(self):
        return np.array([self.temperature])

    def set_self_param_values(self, flattened_params):
        self.temperature = flattened_params[0]
