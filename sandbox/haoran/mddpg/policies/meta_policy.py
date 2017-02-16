import tensorflow as tf
import numpy as np
from rllab.misc.overrides import overrides

from sandbox.haoran.mddpg.policies.nn_policy import FeedForwardPolicy
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy

class MetaPolicy(FeedForwardPolicy):
    """
    A meta policy outputs the samples of a stochastic policy
    """
    def __init__(
            self,
            subpolicy,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.subpolicy = subpolicy
        super().__init__(**kwargs)

    @overrides
    def create_network(self):
        assert isinstance(self.subpolicy, StochasticNNPolicy)
        assert self.action_dim == self.subpolicy._sample_dim
        self.subpolicy_params = self.subpolicy.get_param_values()
        self.subpolicy_params_reassigned = False


        self.subpolicy_original_obs_pl = self.subpolicy.observations_placeholder
        self.subpolicy_original_sample_pl = self.subpolicy._sample_pl

        self.subpolicy_sample_input = super().create_network()
        # self.subpolicy_sample_input = tf.random_normal(
        #     (1, self.subpolicy._sample_dim),
        #     mean=0.,
        #     stddev=1.,
        # )
        self.subpolicy.observations_placeholder = self.observations_placeholder
        return self.subpolicy.create_network(
            sample_input=self.subpolicy_sample_input
        )

    def get_samples(self, observations):
        return self.sess.run(
            self.subpolicy_sample_input,
            {
                self.observations_placeholder: observations,
            }
        )

    def get_sample(self, observation):
        return self.get_samples(np.array([observation]))[0]

    def get_actions_from_samples(self, observations, samples):
        return self.sess.run(
            self.subpolicy.output,
            {
                self.subpolicy_original_obs_pl: observations,
                self.subpolicy_original_sample_pl: samples,
            }
        )
    def get_action_from_sample(self, observation, sample):
        return self.get_actions_from_samples(
            observations=np.array([observation]),
            samples=np.array([sample]),
        )[0]

    @overrides
    def reset(self):
        """
        Hack: since the subpolicy params are modified by
            global_variables_initializer() in DDPG, we should wait until when
            the policy is called and then copy the params back.
        """
        if not self.subpolicy_params_reassigned:
            self.subpolicy.set_param_values(self.subpolicy_params)
            self.subpolicy_params_reassigned = True

    # @overrides
    # def get_action(self, observation):
    #     """ just for debugging """
    #     action = self.sess.run(
    #         self.subpolicy.output,
    #         feed_dict={
    #             self.subpolicy_original_obs_pl: np.array([observation]),
    #             self.subpolicy_original_sample_pl: self.subpolicy._get_input_samples(1),
    #         }
    #     )
    #     return action, {}

from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
class MetaExplorationStrategy(ExplorationStrategy, Serializable):
    def __init__(self, env_spec, substrategy):
        Serializable.quick_init(self, locals())
        self.substrategy = substrategy

    @overrides
    def reset(self):
        self.substrategy.reset()

    @overrides
    def get_action(self, t, observation, policy, **kwargs):
        assert isinstance(policy, MetaPolicy)
        sample = policy.get_sample(observation)
        modified_sample = self.substrategy.get_modified_action(t, sample)
        modified_action = policy.get_action_from_sample(
            observation,
            modified_sample,
        )
        return modified_action

# --------------------------------------------------------------------------
def test():
    from sandbox.haoran.myscripts.retrainer import Retrainer
    from sandbox.haoran.mddpg.gaussian_strategy import GaussianStrategy
    import numpy as np
    import time
    retrainer = Retrainer(
        exp_prefix="tuomas/vddpg/",
        exp_name="exp-000_20170212_232051_458883_tuomas_ant",
        snapshot_file="itr_399.pkl",
        configure_script="",
    )
    retrainer.reload_snapshot()
    env = retrainer.algo.env
    policy = MetaPolicy(
        scope_name="meta_actor",
        observation_dim=env.observation_space.flat_dim,
        action_dim=env.action_space.flat_dim,
        output_nonlinearity=tf.nn.tanh,
        observation_hidden_sizes=(10, 10),
        subpolicy=retrainer.algo.policy,
    )

    # initialize meta policy params and then copy back subpolicy params
    sess = retrainer.sess
    sess.run(tf.global_variables_initializer())
    policy.reset()

    es = MetaExplorationStrategy(
        env_spec=env.spec,
        substrategy=GaussianStrategy(
            env_spec=env.spec,
            mu=0.,
            sigma=1.,
        )
    )

    # rollout
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
            print(r)
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
