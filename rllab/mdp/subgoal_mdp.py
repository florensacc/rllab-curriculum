from rllab.mdp.proxy_mdp import ProxyMDP
from rllab.core.serializable import Serializable
from rllab.misc.ext import AttrDict, flatten_shape_dim


class SubgoalMDP(ProxyMDP, Serializable):

    def __init__(
            self,
            mdp,
            n_goals):
        super(SubgoalMDP, self).__init__(mdp)
        Serializable.quick_init(self, locals())
        self._high_mdp = AttrDict(
            observation_shape=mdp.observation_shape,
            observation_dtype=mdp.observation_dtype,
            action_dim=n_goals,
            action_dtype='uint8'
        )
        obs_dim = flatten_shape_dim(mdp.observation_shape)
        self._low_mdp = AttrDict(
            observation_shape=(obs_dim + n_goals,),
            observation_dtype=mdp.observation_dtype,
            action_dim=mdp.action_dim,
            action_dtype=mdp.action_dtype,
        )

    @property
    def high_mdp(self):
        return self._high_mdp

    @property
    def low_mdp(self):
        return self._low_mdp
