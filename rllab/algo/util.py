import numpy as np
import theano
import time
from rllab.core.serializable import Serializable
from rllab.misc.ext import extract

floatX = theano.config.floatX


def center_advantages(advantages):
    return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)


class ReplayPool(Serializable):
    """
    A utility class for experience replay.
    The code is adapted from https://github.com/spragunr/deep_q_rl
    """

    def __init__(
            self,
            state_shape,
            action_dim,
            max_steps,
            state_dtype=floatX,
            action_dtype=floatX,
            concat_states=False,
            concat_length=1,
            rng=None):
        """Construct a ReplayPool.

        Arguments:
            state_shape - tuple indicating the shape of the state
            action_dim - dimension of the action
            size - capacity of the replay pool
            state_dtype - ...
            action_dtype - ...
            concat_states - whether to concat the past few states as a single
            state, to ensure the Markov property
            concat_length - length of the concatenation
        """

        self.state_shape = state_shape
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.states = np.zeros((max_steps,) + state_shape, dtype=state_dtype)
        self.actions = np.zeros((max_steps, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_steps,), dtype=floatX)
        self.terminal = np.zeros((max_steps,), dtype='bool')
        self.concat_states = concat_states
        self.concat_length = concat_length
        self.state_dtype = state_dtype
        self.action_dtype = action_dtype
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

        if not concat_states:
            assert concat_length == 1, \
                "concat_length must be set to 1 if not concatenating states"

        self.bottom = 0
        self.top = 0
        self.size = 0
        super(ReplayPool, self).__init__(
            self, state_shape, action_dim, max_steps, state_dtype,
            action_dtype, concat_states, concat_length, rng
        )

    def __getstate__(self):
        d = super(ReplayPool, self).__getstate__()
        d["bottom"] = self.bottom
        d["top"] = self.top
        d["size"] = self.size
        d["states"] = self.states
        d["actions"] = self.actions
        d["rewards"] = self.rewards
        d["terminal"] = self.terminal
        d["rng"] = self.rng
        return d

    def __setstate__(self, d):
        super(ReplayPool, self).__setstate__(d)
        self.bottom, self.top, self.size, self.states, self.actions, \
            self.rewards, self.terminal, self.rng = extract(
                d,
                "bottom", "top", "size", "states", "actions", "rewards",
                "terminal", "rng"
            )

    def add_sample(self, state, action, reward, terminal):
        """Add a time step record.

        Arguments:
            state -- current state (or observation)
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended after this
            time step
        """
        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.concat_length)

    def last_concat_state(self):
        """
        Return the most recent sample (concatenated states if needed).
        """
        if self.concat_states:
            indexes = np.arange(self.top - self.concat_length, self.top)
            return self.states.take(indexes, axis=0, mode='wrap')
        else:
            return self.states[self.top - 1]

    def concat_state(self, state):
        """Return a concatenated state, using the last concat_length -
        1, plus state.

        """
        if self.concat_states:
            indexes = np.arange(self.top - self.concat_length + 1, self.top)

            concat_state = np.empty(
                (self.concat_length,) + self.state_shape,
                dtype=floatX
            )
            concat_state[0:self.concat_length - 1] = \
                self.states.take(indexes, axis=0, mode='wrap')
            concat_state[-1] = state
            return concat_state
        else:
            return state

    def random_batch(self, batch_size):
        """
        Return corresponding states, actions, rewards, terminal status, and
next_states for batch_size randomly chosen state transitions.
        """
        # Allocate the response.

        states = np.zeros(
            (batch_size, self.concat_length) + self.state_shape,
            dtype=self.state_dtype
        )
        actions = np.zeros(
            (batch_size, self.action_dim),
            dtype=self.action_dtype
        )
        rewards = np.zeros((batch_size,), dtype=floatX)
        terminal = np.zeros((batch_size,), dtype='bool')
        next_states = np.zeros(
            (batch_size, self.concat_length) + self.state_shape,
            dtype=self.state_dtype
        )

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(
                self.bottom,
                self.bottom + self.size - self.concat_length
            )

            initial_indices = np.arange(index, index + self.concat_length)
            transition_indices = initial_indices + 1
            end_index = index + self.concat_length - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but the last frame may be
            # terminal. If the last frame of the initial state is
            # terminal, then the last frame of the transitioned state
            # will actually be the first frame of a new episode, which
            # the Q learner recognizes and handles correctly during
            # training by zeroing the discounted future reward estimate.
            if np.any(self.terminal.take(initial_indices[0:-1], mode='wrap')):
                continue

            # Add the state transition to the response.
            states[count] = self.states.take(
                initial_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')
            next_states[count] = self.states.take(
                transition_indices, axis=0, mode='wrap')
            count += 1

        if not self.concat_states:
            # If we're not concatenating states, we should squeeze the second
            # dimension in states and next_states
            states = np.squeeze(states, axis=1)
            next_states = np.squeeze(next_states, axis=1)

        return states, actions, rewards, next_states, terminal


# TESTING CODE BELOW THIS POINT...

def simple_tests():
    np.random.seed(222)
    dataset = ReplayPool(
        state_shape=(3, 2),
        action_dim=1,
        max_steps=6,
        concat_states=True,
        concat_length=4
    )
    for _ in range(10):
        img = np.random.randint(0, 256, size=(3, 2))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        print 'img', img
        dataset.add_sample(img, action, reward, terminal)
        print "S", dataset.states
        print "A", dataset.actions
        print "R", dataset.rewards
        print "T", dataset.terminal
        print "SIZE", dataset.size
        print
    print "LAST CONCAT STATE", dataset.last_concat_state()
    print
    print 'BATCH', dataset.random_batch(2)


def speed_tests():

    dataset = ReplayPool(
        state_shape=(80, 80),
        action_dim=1,
        max_steps=20000,
        concat_states=True,
        concat_length=4,
    )

    img = np.random.randint(0, 256, size=(80, 80))
    action = np.random.randint(16)
    reward = np.random.random()
    start = time.time()
    for _ in range(100000):
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset.add_sample(img, action, reward, terminal)
    print "samples per second: ", 100000 / (time.time() - start)

    start = time.time()
    for _ in range(200):
        dataset.random_batch(32)
    print "batches per second: ", 200 / (time.time() - start)

    print dataset.last_concat_state()


def trivial_tests():

    dataset = ReplayPool(
        state_shape=(1, 2),
        action_dim=1,
        max_steps=3,
        concat_states=True,
        concat_length=2
    )

    img1 = np.array([[1, 1]], dtype='uint8')
    img2 = np.array([[2, 2]], dtype='uint8')
    img3 = np.array([[3, 3]], dtype='uint8')

    dataset.add_sample(img1, 1, 1, False)
    dataset.add_sample(img2, 2, 2, False)
    dataset.add_sample(img3, 2, 2, True)
    print "last", dataset.last_concat_state()
    print "random", dataset.random_batch(1)


def max_size_tests():
    dataset1 = ReplayPool(
        state_shape=(4, 3),
        action_dim=1,
        max_steps=10,
        concat_states=True,
        concat_length=4,
        rng=np.random.RandomState(42)
    )
    dataset2 = ReplayPool(
        state_shape=(4, 3),
        action_dim=1,
        max_steps=1000,
        concat_states=True,
        concat_length=4,
        rng=np.random.RandomState(42)
    )
    for _ in range(100):
        img = np.random.randint(0, 256, size=(4, 3))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset1.add_sample(img, action, reward, terminal)
        dataset2.add_sample(img, action, reward, terminal)
        np.testing.assert_array_almost_equal(dataset1.last_concat_state(),
                                             dataset2.last_concat_state())
        print "passed"


def test_memory_usage_ok():
    import memory_profiler
    dataset = ReplayPool(
        state_shape=(80, 80),
        action_dim=1,
        max_steps=100000,
        concat_states=True,
        concat_length=4
    )
    last = time.time()

    for i in xrange(1000000000):
        if (i % 100000) == 0:
            print i
        dataset.add_sample(np.random.random((80, 80)), 1, 1, False)
        if i > 200000:
            dataset.random_batch(32)
        if (i % 10007) == 0:
            print time.time() - last
            mem_usage = memory_profiler.memory_usage(-1)
            print len(dataset), mem_usage
        last = time.time()


def main():
    speed_tests()
    # test_memory_usage_ok()
    max_size_tests()
    simple_tests()

if __name__ == "__main__":
    main()
