from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.hrl.bonus_evaluators.base import BonusEvaluator
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.envs.base import EnvSpec
from rllab.spaces.discrete import Discrete
from rllab.misc import logger
from rllab.misc import tensor_utils
from rllab.core.serializable import Serializable
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
import numpy as np


class MarginalParsimonyBonusEvaluator(BonusEvaluator, Serializable):
    """
    Compute the penalty (negative bonus) as I(St; At). Furthermore, we assume that H(At) is constant. Hence the bonus
    should be H(At|St). We use a variational approximation and fit a parametric distribution q(at|st), and compute the
    bonus reward as -log(q(at|st)).

    This simply reduces to maximum entropy, but in a marginalized manner. Normally the policy is recurrent and it's
    intractable to compute the exact, marginalized entropy.
    """

    def __init__(self, env_spec, policy, bonus_coeff=1., regressor_cls=None, regressor_args=None):
        """
        :type env_spec: EnvSpec
        :type policy: StochasticGRUPolicy
        :type regressor_cls: type
        :type regressor_args: dict
        """
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.bonus_coeff = bonus_coeff
        self.policy = policy
        assert isinstance(env_spec.action_space, Discrete)
        if regressor_cls is None:
            regressor_cls = CategoricalMLPRegressor
        if regressor_args is None:
            regressor_args = dict()
        self.regressor = regressor_cls(
            input_shape=(env_spec.observation_space.flat_dim,),
            output_dim=env_spec.action_space.n,
            name="p(at|st)",
            **regressor_args
        )

    def fit(self, paths):
        obs = np.concatenate([p["observations"] for p in paths])
        actions = np.concatenate([p["actions"] for p in paths])
        self.regressor.fit(obs, actions)

    def predict(self, path):
        obs = path["observations"]
        actions = path["actions"]
        return - self.bonus_coeff * self.regressor.predict_log_likelihood(obs, actions)

    def log_diagnostics(self, paths):
        obs = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        ent_a_given_s = np.mean(-self.regressor.predict_log_likelihood(obs, actions))
        logger.record_tabular("approx_H(a|s)", ent_a_given_s)
        # we'd also like to compute I(a,h|s) = H(a|s) - H(a|h,s)
        ent_a_given_hs = np.mean(self.policy.action_dist.entropy(dict(prob=agent_infos["action_prob"])))
        logger.record_tabular("approx_I(a;h|s)", ent_a_given_s - ent_a_given_hs)
