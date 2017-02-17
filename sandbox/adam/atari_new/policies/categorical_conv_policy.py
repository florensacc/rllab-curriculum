
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy as BaseCategoricalConvPolicy
from rllab.misc.overrides import overrides


class CategoricalConvPolicy(BaseCategoricalConvPolicy):

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        # actions = list(map(self.action_space.weighted_sample, probs))
        actions = self.action_space.weighted_sample_n(probs)  # requires space from adam.sandbox.atari_new.spaces.discrete
        return actions, dict(prob=probs)
