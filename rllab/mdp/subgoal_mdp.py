from rllab.mdp.proxy_mdp import ProxyMDP
from rllab.mdp.mdp_spec import MDPSpec
from rllab.core.serializable import Serializable
from rllab.misc.ext import AttrDict, flatten_shape_dim


class SubgoalMDP(ProxyMDP, Serializable):

    def __init__(
            self,
            mdp,
            n_subgoals):
        super(SubgoalMDP, self).__init__(mdp)
        Serializable.quick_init(self, locals())
        self._high_mdp = MDPSpec(
            observation_shape=mdp.observation_shape,
            observation_dtype=mdp.observation_dtype,
            action_dim=n_subgoals,
            action_dtype='uint8',
        )
        self._low_mdp = MDPSpec(
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
    def high_mdp(self):
        return self._high_mdp

    @property
    def low_mdp(self):
        return self._low_mdp
