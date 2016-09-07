import theano
import numpy as np


class ReplayPool(object):
    """Replay pool"""

    def __init__(
            self,
            max_pool_size,
            observation_shape,
            action_dim,
            observation_dtype=theano.config.floatX,
            action_dtype=theano.config.floatX,
            num_seq_frames=1,
            **kwargs
    ):
        self._observation_shape = observation_shape
        self._action_dim = action_dim
        self._observation_dtype = observation_dtype
        self._action_dtype = action_dtype
        self._max_pool_size = max_pool_size
        self._num_seq_frames = num_seq_frames

        self._observations = np.zeros(
            (max_pool_size,) + observation_shape,
            dtype=observation_dtype
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
            dtype=action_dtype
        )
        self._rewards = np.zeros(max_pool_size, dtype='float32')
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def __str__(self):
        sb = []
        for key in self.__dict__:
            sb.append(
                "{key}='{value}'".format(key=key, value=self.__dict__[key]))
        return ', '.join(sb)

    def add_sample(self, observation, action, reward, terminal):
        """Add sample to replay pool."""
        # Select last frame, which is the 'true' frame. Only add this one, to save replay pool memory. When
        # the samples are fetched, we rebuild the sequence.
        self._observations[self._top] = observation[-self._observation_shape[0]:]
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        """Retrieve random batch from replay pool."""
        # Here, based on the num_seq_frames, we will construct a batch of elements that comform num_seq_frames.
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count, arr_obs = 0, None
        while count < batch_size:
            # We don't want to hit the bottom when using num_seq_frames in history.
            index = np.random.randint(
                self._bottom + self._num_seq_frames, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample. Also check whether terminal sample: no next state exists.
            if (index == self._size - 1 and self._size <= self._max_pool_size) or self._terminals[index] == 1:
                continue
            transition_index = (index + 1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            # Here we add num_seq_frames - 1 additional previous frames; until we encounter term frame, in which
            # case we add black frames.
            lst_obs = [None] * self._num_seq_frames
            insert_empty = np.zeros(batch_size, dtype='bool')
            for i in xrange(self._num_seq_frames):
                obs = self._observations[indices - i]
                insert_empty = np.maximum(self._terminals[indices - i].astype('bool'), insert_empty)
                if insert_empty.any():
                    obs_prev = self._observations[indices - i + 1]
                    obs[insert_empty] = obs_prev[insert_empty]
                lst_obs[self._num_seq_frames - i - 1] = obs
            arr_obs = np.stack(lst_obs, axis=1).reshape((lst_obs[0].shape[0], -1))

            count += 1
        return dict(
            observations=arr_obs,
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size
