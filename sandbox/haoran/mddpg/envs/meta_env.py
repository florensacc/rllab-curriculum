import numpy as np
import tensorflow as tf
from scipy.stats import norm

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from rllab.envs.normalized_env import normalize

from sandbox.haoran.myscripts import tf_utils
from sandbox.tuomas.mddpg.policies.stochastic_policy \
    import StochasticNNPolicy as TfStochasticNNPolicy
from sandbox.haoran.mddpg.policies.stochastic_policy_theano \
    import StochasticNNPolicy as TheanoStochasticNNPolicy

class MetaEnv(ProxyEnv, Serializable):
    """
    An environent wrapper that transforms action inputs into noise of a
        stochastic policy, which outputs the actual actions in the wrapped
        environment.
    """
    def __init__(
        self,
        env,
        transform_policy,
    ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        assert isinstance(transform_policy, TfStochasticNNPolicy)
        self.transform_policy = transform_policy
        self.Ds = transform_policy._sample_dim
        confidence = 0.9 # total prob of the bounded action space
        confidence_1d = np.power(confidence, 1./self.Ds)
            # prob of each dim (assuming a hyper-cubic action space)
        self.action_bound = norm.ppf(1. - 0.5 * (1. - confidence_1d))
        self.transform_policy_params_reassigned = False
        self.transform_policy_params = self.transform_policy.get_param_values()

    def get_default_sigma_after_normalization(self):
        return 1./ self.action_bound

    @overrides
    def reset(self, **kwargs):
        self.sess = tf.get_default_session() or tf_utils.create_session()
        if not self.transform_policy_params_reassigned:
            self.transform_policy.set_param_values(
                self.transform_policy_params)
            self.transform_policy_params_reassigned = True
        return self._wrapped_env.reset(**kwargs)

    @overrides
    def get_param_values(self):
        params = self._wrapped_env.get_param_values()
        params["transform_policy_params"] = self.transform_policy_params
        return params

    @overrides
    def set_param_values(self, params):
        # hacky: only tested for parallel workers
        self.transform_policy_params = params["transform_policy_params"]
        self.transform_policy_params_reassigned = False
        self._wrapped_env.set_param_values(params)
            # note: better to pop "transform_policy_params"

    @property
    @overrides
    def action_space(self):
        ub = self.action_bound * np.ones(self.Ds)
        return spaces.Box(-1 * ub, ub)

    @overrides
    def step(self, action):
        print(action)
        # Use the line below to test whether parallel worker have the correct
        # transform_policy params
        # print(np.linalg.norm(self.transform_policy.get_param_values()))

        if action.ndim == 1:
            action = np.array([action])
        obs = self._wrapped_env.get_current_obs()
        if obs.ndim == 1:
            obs = np.array([obs])

        transformed_action = self.sess.run(
            self.transform_policy.output,
            {
                self.transform_policy.observations_placeholder: obs,
                self.transform_policy._sample_pl: action,
            }
        )
        return self._wrapped_env.step(transformed_action)

class MetaEnvTheano(ProxyEnv, Serializable):
    """
    An environent wrapper that transforms action inputs into noise of a
        stochastic policy, which outputs the actual actions in the wrapped
        environment.
    """
    def __init__(
        self,
        env,
        transform_policy,
    ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        assert isinstance(transform_policy, TheanoStochasticNNPolicy)
        self.transform_policy = transform_policy
        self.Ds = transform_policy._sample_dim
        confidence = 0.9 # total prob of the bounded action space
        confidence_1d = np.power(confidence, 1./self.Ds)
            # prob of each dim (assuming a hyper-cubic action space)
        self.action_bound = norm.ppf(1. - 0.5 * (1. - confidence_1d))
        self.transform_policy_params_reassigned = False
        self.transform_policy_params = self.transform_policy.get_param_values()

    def get_default_sigma_after_normalization(self):
        return 1./ self.action_bound

    @overrides
    def reset(self, **kwargs):
        if not self.transform_policy_params_reassigned:
            self.transform_policy.set_param_values(
                self.transform_policy_params)
            self.transform_policy_params_reassigned = True
        return self._wrapped_env.reset(**kwargs)

    @overrides
    def get_param_values(self):
        params = self._wrapped_env.get_param_values()
        params["transform_policy_params"] = self.transform_policy_params
        return params

    @overrides
    def set_param_values(self, params):
        # hacky: only tested for parallel workers
        self.transform_policy_params = params["transform_policy_params"]
        self.transform_policy_params_reassigned = False
        self._wrapped_env.set_param_values(params)
            # note: better to pop "transform_policy_params"

    @property
    @overrides
    def action_space(self):
        ub = self.action_bound * np.ones(self.Ds)
        return spaces.Box(-1 * ub, ub)

    @overrides
    def step(self, action):
        # Use the line below to test whether parallel worker have the correct
        # transform_policy params
        # print(np.linalg.norm(self.transform_policy.get_param_values()))

        transformed_action, info = self.transform_policy.get_action(
            observation=self._wrapped_env.get_current_obs(),
            sample=action
        )
        return self._wrapped_env.step(transformed_action)

# --------------------------------------------------------------------------
def test():
    from sandbox.haoran.myscripts.retrainer import Retrainer
    from sandbox.haoran.mddpg.gaussian_strategy import GaussianStrategy
    from sandbox.haoran.mddpg.policies.nn_policy import FeedForwardPolicy
    import numpy as np
    import time
    retrainer = Retrainer(
        exp_prefix="tuomas/vddpg/",
        exp_name="exp-000_20170212_232051_458883_tuomas_ant",
        snapshot_file="itr_399.pkl",
        configure_script="",
    )
    retrainer.reload_snapshot()
    meta_env = MetaEnv(
        env=retrainer.algo.env,
        transform_policy=retrainer.algo.policy,
    )
    # should normalize the env since the policy output is bounded by [-1, 1]
    env = normalize(meta_env)

    policy = FeedForwardPolicy(
        scope_name="meta_actor",
        observation_dim=env.observation_space.flat_dim,
        action_dim=env.action_space.flat_dim,
        output_nonlinearity=tf.nn.tanh,
        observation_hidden_sizes=(8, 8),
    )
    es = GaussianStrategy(
        env_spec=env.spec,
        mu=0.,
        sigma=meta_env.get_default_sigma(),
    )
    # the noise should have std = 1 in the meta_env by default, since the stoc
    # policy has standard normal noise

    # rollout
    sess = retrainer.sess
    sess.run(tf.global_variables_initializer())
    animated = True
    speedup = 10
    max_path_length = 500

    while True:
        o = env.reset()
        es.reset()
        t = 0
        if animated:
            env.render()
        while t < max_path_length:
            a = es.get_action(t, o, policy)
            next_o, r, d, env_info = env.step(a)
            t += 1
            if d:
                break
            o = next_o
            if animated:
                env.render()
                timestep = 0.05
                time.sleep(timestep / speedup)
        if animated:
            env.render(close=True)


if __name__ == "__main__":
    test()
