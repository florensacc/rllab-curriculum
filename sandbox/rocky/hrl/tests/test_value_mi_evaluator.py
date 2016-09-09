


from sandbox.rocky.hrl.mi_evaluator.state_based_value_mi_evaluator import downsampled_discount_cumsum

from nose2.tools import such
import numpy as np
import scipy.signal
import time
import numba


def fast_discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]


def slow_discount_cumsum(x, discount):
    ret = []
    discount_exps = np.arange(len(x))
    for t in range(len(x)):
        offset_discount_exps = discount_exps[t:] - discount_exps[t]
        offset_x = x[t:]
        discount_return = np.sum((discount ** offset_discount_exps) * offset_x)
        ret.append(discount_return)
    return np.array(ret)


@numba.jit
def numba_discount_cumsum(x, discount):
    result = np.empty_like(x)
    cur_sum = 0
    for t in range(len(x) - 1, -1, -1):
        cur_sum = cur_sum * discount + x[t]
        result[t] = cur_sum
    return result


def slow_downsampled_discount_cumsum(x, discount, subgoal_interval):
    result = []
    discount_exps = np.cast['int'](np.floor(np.arange(len(x)) * 1.0 / subgoal_interval))
    for t in range(len(x)):
        offset_discount_exps = discount_exps[t:] - discount_exps[t]
        offset_rewards = x[t:]
        discount_return = np.sum((discount ** offset_discount_exps) * offset_rewards)
        result.append(discount_return)
    return np.array(result)


@numba.jit
def numba_downsampled_discount_cumsum(x, discount, subgoal_interval):
    result = np.empty_like(x)
    cur_sum = 0
    terminal = len(x) - 1
    for t in range(terminal, -1, -1):
        multiplier = discount if (t % subgoal_interval == subgoal_interval - 1) else 1.
        cur_sum = cur_sum * multiplier + x[t]
        result[t] = cur_sum
    return result


with such.A("State-based value MI evaluator") as it:
    @it.should
    def benchmark_discount_cumsum():
        x = np.random.uniform(size=(100,))
        discount = 0.99
        np.testing.assert_array_almost_equal(fast_discount_cumsum(x, discount), numba_discount_cumsum(x, discount))
        np.testing.assert_array_almost_equal(fast_discount_cumsum(x, discount), slow_discount_cumsum(x, discount))
        start_time = time.time()
        for _ in range(10000):
            fast_discount_cumsum(x, discount)
        time1 = time.time() - start_time
        start_time = time.time()
        for _ in range(1000):
            slow_discount_cumsum(x, discount)
        time2 = time.time() - start_time

        # allow compilation
        start_time = time.time()
        for _ in range(10000):
            numba_discount_cumsum(x, discount)
        time3 = time.time() - start_time

        print("Fast discount_cumsum took %fms" % (time1 / 10))
        print("Slow discount_cumsum took %fms" % (time2))
        print("Numba discount_cumsum took %fms" % (time3 / 10))


    @it.should
    def benchmark_downsampled_discount_cumsum():
        x = np.random.uniform(size=(100,))
        discount = 0.99
        subgoal_interval = 3
        np.testing.assert_array_almost_equal(numba_downsampled_discount_cumsum(x, discount, subgoal_interval),
                                             slow_downsampled_discount_cumsum(x, discount, subgoal_interval))
        np.testing.assert_array_almost_equal(downsampled_discount_cumsum(x, discount, subgoal_interval),
                                             slow_downsampled_discount_cumsum(x, discount, subgoal_interval))
        start_time = time.time()
        for _ in range(1000):
            slow_downsampled_discount_cumsum(x, discount, subgoal_interval)
        time1 = time.time() - start_time
        start_time = time.time()
        for _ in range(100):
            numba_downsampled_discount_cumsum(x, discount, subgoal_interval)
        time2 = time.time() - start_time
        print("Slow downsampled_discount_cumsum took %fms" % (time1 / 1000 * 1000))
        print("Numba downsampled_discount_cumsum took %fms" % (time2 / 100 * 1000))

it.createTests(globals())
