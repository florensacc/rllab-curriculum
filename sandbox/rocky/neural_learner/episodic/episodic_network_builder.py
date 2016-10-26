from sandbox.rocky.tf.spaces import Product, Box, Discrete
import sandbox.rocky.tf.core.layers as L
import sandbox.rocky.analogy.core.layers as LL


class EpisodicNetworkBuilder(object):
    def __init__(self, env_spec):
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space

    def split_obs_layer(self, l_obs):
        # A few sanity checks
        assert isinstance(self.observation_space, Product)
        assert len(self.observation_space.components) == 4
        raw_obs_space, action_space, reward_space, terminal_space = self.observation_space.components
        assert isinstance(reward_space, Box) and reward_space.shape == (1,)
        assert isinstance(terminal_space, Box) and terminal_space.shape == (1,)
        assert isinstance(action_space, Discrete) and action_space.n == self.action_space.n

        raw_obs_dim = raw_obs_space.flat_dim
        obs_dim = self.observation_space.flat_dim
        action_dim = action_space.flat_dim

        assert obs_dim == raw_obs_dim + action_dim + 2

        raw_obs_layer = L.SliceLayer(l_obs, slice(raw_obs_dim), axis=2)
        prev_action_layer = L.SliceLayer(l_obs, slice(raw_obs_dim, raw_obs_dim + action_dim), axis=2)
        reward_layer = L.SliceLayer(l_obs, slice(raw_obs_dim + action_dim, raw_obs_dim + action_dim + 1), axis=2)
        terminal_layer = L.SliceLayer(l_obs, slice(raw_obs_dim + action_dim + 1, raw_obs_dim + action_dim + 2), axis=2)

        return raw_obs_layer, prev_action_layer, reward_layer, terminal_layer

    def new_obs_feature_layer(self, l_obs):
        return l_obs

    def new_action_feature_layer(self, l_action):
        return l_action

    def new_rnn_layer(self, feature_layer, cell):
        return LL.TfRNNLayer(
            incoming=feature_layer,
            cell=cell,
        )
