from sandbox.rocky.neural_learner.envs.parallel_atari_env import NaiveCircularBuffer
import numpy as np


def test_circular_buffer():
    buf_cls = NaiveCircularBuffer

    buffer = buf_cls(batch_size=2, buffer_size=4, data_shape=(), dtype=np.int)
    buffer.push([1, 1])
    buffer.reset([True, False])
    buffer.push([2, 2])
    buffer.push([3, 3])
    buffer.push([4, 4])
    result = buffer.last(4)
    np.testing.assert_allclose(result, np.array([[0, 2, 3, 4], [1, 2, 3, 4]]))
    buffer.reset([False, True])
    buffer.push([5, 5])
    result = buffer.last(3)
    np.testing.assert_allclose(result, np.array([[3, 4, 5], [0, 0, 5]]))


if __name__ == "__main__":
    test_circular_buffer()
