from rllab.mdp.proxy_mdp import ProxyMDP
from rllab.mdp.mdp_spec import MDPSpec
from rllab.core.serializable import Serializable


class SubgoalMDP(ProxyMDP, Serializable):

    def __init__(
            self,
            mdp,
            n_subgoals):
        super(SubgoalMDP, self).__init__(mdp)
        Serializable.quick_init(self, locals())
        self._high_mdp_spec = MDPSpec(
            observation_shape=mdp.observation_shape,
            observation_dtype=mdp.observation_dtype,
            action_dim=n_subgoals,
            action_dtype='uint8',
        )
        self._low_mdp_spec = MDPSpec(
            observation_shape=(mdp.observation_dim + n_subgoals,),
            observation_dtype=mdp.observation_dtype,
            action_dim=mdp.action_dim,
            action_dtype=mdp.action_dtype,
        )
        self._n_subgoals = n_subgoals

    @property
    def n_subgoals(self):
        return self._n_subgoals

    @property
    def high_mdp_spec(self):
        return self._high_mdp_spec

    @property
    def low_mdp_spec(self):
        return self._low_mdp_spec

    @property
    def spec(self):
        return SubgoalMDPSpec(
            observation_shape=self._mdp.observation_shape,
            observation_dtype=self._mdp.observation_dtype,
            action_dim=self._mdp.action_dim,
            action_dtype=self._mdp,
            n_subgoals=self._n_subgoals,
            high_mdp_spec=self.high_mdp_spec,
            low_mdp_spec=self.low_mdp_spec
        )


class SubgoalMDPSpec(MDPSpec):

    def __init__(
            self,
            observation_shape,
            observation_dtype,
            action_dim,
            action_dtype,
            n_subgoals,
            high_mdp_spec,
            low_mdp_spec):
        Serializable.quick_init(self, locals())
        super(SubgoalMDPSpec, self).__init__(
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            action_dim=action_dim,
            action_dtype=action_dtype,
        )
        self._n_subgoals = n_subgoals
        self._high_mdp_spec = high_mdp_spec
        self._low_mdp_spec = low_mdp_spec

    @property
    def n_subgoals(self):
        return self._n_subgoals

    @property
    def high_mdp_spec(self):
        return self._high_mdp_spec

    @property
    def low_mdp_spec(self):
        return self._low_mdp_spec
