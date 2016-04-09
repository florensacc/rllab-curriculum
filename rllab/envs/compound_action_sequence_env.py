from .proxy_env import ProxyEnv
from rllab.core.serializable import Serializable
from rllab.spaces import Discrete
from rllab.envs.base import Step
from rllab.spaces import Product
import numpy as np
from rllab.misc import special


class CompoundActionSequenceEnv(ProxyEnv, Serializable):
    """
    Takes a discrete action mdp, and turns it into an mdp with compound actions so that each original action is
    mapped from a sequence of discrete actions. An invalid sequence will simply map to nothing in the original MDP.

    To succeed in this mdp, the agent needs to learn the low-level action sequences required to trigger actions in the
    original mdp, and perform exploration using these learned action sequence primitives (thus, a hierarchy).
    """

    def __init__(self, wrapped_env, action_map, action_dim=None, reset_history=True, obs_include_history=False):
        """
        Constructs a compound mdp.
        :param mdp: The original mdp.
        :param action_map: The map from each action in the original mdp to a sequence of actions, required to trigger
        the corresponding action in the original mdp
        :param action_dim: The action dimension of the compound mdp. By default this is the same as the original mdp
        :param reset_history: This only works if all the action sequences are of the same length. This flag controls
        whether the history will be cleared after it reaches the action sequence length, whether it matches an
        original action or not.
        :return:
        """
        Serializable.quick_init(self, locals())
        super(CompoundActionSequenceEnv, self).__init__(wrapped_env)
        assert isinstance(wrapped_env.action_space, Discrete), \
            "Expected Discrete action space but got %s" % str(wrapped_env.action_space)
        assert len(action_map) == wrapped_env.action_space.n
        action_strs = [",".join(map(str, x)) for x in action_map]
        # ensure no duplicates
        assert len(set(action_strs)) == len(action_strs)
        # ensure that no action sequence is a prefix or suffix of the other
        assert not any([x.startswith(y) for x in action_strs for y in action_strs if x != y])
        assert not any([x.endswith(y) for x in action_strs for y in action_strs if x != y])
        if reset_history:
            assert len(set([len(x) for x in action_map])) == 1
        self._action_map = map(np.array, action_map)
        self._action_history = []
        if action_dim is None:
            self._action_dim = wrapped_env.action_space.n
        else:
            self._action_dim = action_dim
        self._raw_obs = None
        self._reset_history = reset_history
        self._obs_include_history = obs_include_history
        self._history_length = len(action_map[0])
        if obs_include_history:
            assert reset_history
            # observation dimension is action_dim+1 to account for empty slots
            self._observation_space = Product(
                [self.wrapped_env.observation_space] + [Discrete(self._action_dim + 1) for _ in
                                                        xrange(self._history_length)]
            )
        else:
            self._observation_space = self.wrapped_env.observation_space
        self._action_space = Discrete(self._action_dim)
        self.wrapped_env.reset()

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        obs = self.wrapped_env.reset()
        self._action_history = []
        self._raw_obs = obs
        return self._get_current_obs()

    def _get_current_obs(self):
        if self._obs_include_history:
            # If history_length=5, action_dim=4, and action_history = [1, 2, 3], then the observation would include a
            # vector of the form [3, 2, 1, 4, 4]. Note that the most recent action appears first, and any empty
            # slots are padded at the end
            # If the action history is longer than history_length, then only the last few will be included
            included = self._action_history[::-1][:self._history_length]
            # make shape checking happy
            # one_hots = np.array([special.to_onehot(x, self._action_dim) for x in included])
            padded = included + [self._action_dim] * (self._history_length - len(included))
            return (self._raw_obs,) + tuple(padded)
        else:
            return self._raw_obs

    @property
    def observation_space(self):
        return self._observation_space

    def step(self, action):
        self._action_history.append(action)
        # check if the last few actions match any real action
        real_action = None
        for idx, action_list in enumerate(self._action_map):
            if np.array_equal(action_list, self._action_history[-len(action_list):]):
                real_action = idx
                break
        if real_action is not None:
            next_raw_obs, reward, done, _ = self.wrapped_env.step(real_action)
            self._raw_obs = next_raw_obs
            # clear the action history so far
            self._action_history = []
        else:
            reward = 0
            done = False
            if len(self._action_history) >= len(self._action_map[0]) and self._reset_history:
                self._action_history = []
        return Step(observation=self._get_current_obs(), reward=reward, done=done)
