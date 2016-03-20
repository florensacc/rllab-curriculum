from rllab.mdp.proxy_mdp import ProxyMDP
from rllab.core.serializable import Serializable
import numpy as np
from rllab.misc import special


class CompoundActionSequenceMDP(ProxyMDP, Serializable):

    """
    Takes a discrete action mdp, and turns it into an mdp with compound actions so that each original action is
    mapped from a sequence of discrete actions. An invalid sequence will simply map to nothing in the original MDP.

    To succeed in this mdp, the agent needs to learn the low-level action sequences required to trigger actions in the
    original mdp, and perform exploration using these learned action sequence primitives (thus, a hierarchy).
    """

    def __init__(self, mdp, action_map, action_dim=None):
        """
        Constructs a compound mdp.
        :param mdp: the original mdp.
        :param action_map: the map from each action in the original mdp to a sequence of actions, required to trigger
        the corresponding action in the original mdp
        :param action_dim: the action dimension of the compound mdp. By default this is the same as the original mdp
        :return:
        """
        Serializable.quick_init(self, locals())
        super(CompoundActionSequenceMDP, self).__init__(mdp)
        assert len(action_map) == mdp.action_dim
        action_strs = [",".join(map(str, x)) for x in action_map]
        # ensure no duplicates
        assert len(set(action_strs)) == len(action_strs)
        # ensure that no action sequence is a prefix or suffix of the other
        assert not any([x.startswith(y) for x in action_strs for y in action_strs if x != y])
        assert not any([x.endswith(y) for x in action_strs for y in action_strs if x != y])
        self._action_map = map(np.array, action_map)
        # self._action_strs = action_strs
        self._action_history = []
        if action_dim is None:
            self._action_dim = self._mdp.action_dim
        else:
            self._action_dim = action_dim
        self._obs = None
        self._mdp.reset()

    @property
    def action_dim(self):
        return self._action_dim

    def reset(self):
        obs = self._mdp.reset()
        self._action_history = []
        self._obs = obs
        return obs

    def step(self, action):
        self._action_history.append(special.from_onehot(action))
        # check if the last few actions match any real action
        real_action = None
        for idx, action_list in enumerate(self._action_map):
            if np.array_equal(action_list, self._action_history[-len(action_list):]):
                real_action = idx
                break
        if real_action is not None:
            next_obs, reward, done = self._mdp.step(special.to_onehot(real_action, self._mdp.action_dim))
            self._obs = next_obs
            # clear the action history so far
            self._action_history = []
        else:
            next_obs = self._obs
            reward = 0
            done = False
        return next_obs, reward, done
