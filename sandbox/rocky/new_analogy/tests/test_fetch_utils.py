def test_circular_queue():
    from sandbox.rocky.new_analogy.fetch_utils import CircularQueue
    import numpy as np
    queue = CircularQueue(max_size=10, data_shape=(1,))
    queue.extend(np.arange(20).reshape((-1, 1)))
    np.testing.assert_allclose(queue.data, np.arange(10, 20).reshape((-1, 1)))
    assert queue.size == 10
    assert queue.head == 0
    queue.extend(np.arange(5).reshape((-1, 1)))
    np.testing.assert_allclose(queue.data[:5], np.arange(5).reshape((-1, 1)))
    np.testing.assert_allclose(queue.data[5:], np.arange(15, 20).reshape((-1, 1)))
    assert queue.size == 10
    assert queue.head == 5


def test_discretize_actions():
    from sandbox.rocky.new_analogy.fetch_utils import DiscretizedRelativeFetchPolicy
    import numpy as np

    pol = DiscretizedRelativeFetchPolicy(rel_policy=None, disc_intervals=np.asarray([
        [-0.05, -0.01, -0.001, 0, 0.001, 0.01, 0.05],
        [-0.05, -0.01, -0.001, 0, 0.001, 0.01, 0.05],
        [-0.05, -0.01, -0.001, 0, 0.001, 0.01, 0.05],
    ]))
    assert pol.discretize_actions(np.asarray([[-1, 0, 0]], dtype=np.float))[0, 0] == -0.05
    assert pol.discretize_actions(np.asarray([[-0.05, 0, 0]], dtype=np.float))[0, 0] == -0.05
    assert pol.discretize_actions(np.asarray([[-0.02, 0, 0]], dtype=np.float))[0, 0] == -0.01
    assert pol.discretize_actions(np.asarray([[-0.04, 0, 0]], dtype=np.float))[0, 0] == -0.01
    assert pol.discretize_actions(np.asarray([[0, 0, 0]], dtype=np.float))[0, 0] == 0
    assert pol.discretize_actions(np.asarray([[0.0001, 0, 0]], dtype=np.float))[0, 0] == 0.001
    assert pol.discretize_actions(np.asarray([[0.001, 0, 0]], dtype=np.float))[0, 0] == 0.001
    assert pol.discretize_actions(np.asarray([[0.0012, 0, 0]], dtype=np.float))[0, 0] == 0.001
    assert pol.discretize_actions(np.asarray([[0.007, 0, 0]], dtype=np.float))[0, 0] == 0.001
    assert pol.discretize_actions(np.asarray([[0.013, 0, 0]], dtype=np.float))[0, 0] == 0.01
