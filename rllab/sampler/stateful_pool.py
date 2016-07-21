from __future__ import print_function
from __future__ import absolute_import
import multiprocessing as mp
from rllab.misc import logger
import pyprind
import time
import traceback
import sys
import atexit
import Queue


class ProgBarCounter(object):
    def __init__(self, total_count):
        self.total_count = total_count
        self.max_progress = 1000000
        self.cur_progress = 0
        self.cur_count = 0
        if not logger.get_log_tabular_only():
            self.pbar = pyprind.ProgBar(self.max_progress)
        else:
            self.pbar = None

    def inc(self, increment):
        if not logger.get_log_tabular_only():
            self.cur_count += increment
            new_progress = self.cur_count * self.max_progress / self.total_count
            if new_progress < self.max_progress:
                self.pbar.update(new_progress - self.cur_progress)
            self.cur_progress = new_progress

    def stop(self):
        if not logger.get_log_tabular_only():
            self.pbar.stop()


class SharedGlobal(object):
    pass


def worker_main_loop(worker_id, G, worker_read_queue, worker_write_queue, master_read_queue, master_write_queue):
    try:
        G.worker_id = worker_id
        while True:
            msg = worker_read_queue.get()
            if hasattr(msg, '__len__'):
                task = msg[0]
                if task == 'run':
                    try:
                        runner = msg[1]
                        args = msg[2]
                        result = runner(G, *args)
                        worker_write_queue.put(('success', result))
                    except Exception as e:
                        worker_write_queue.put(('error', e))
                        # raise
                elif task == 'terminate':
                    break
                elif task == 'run_map':
                    # switch to query from master_queue
                    while True:
                        msg = master_read_queue.get()
                        task = msg[0]
                        if task == 'run_map':
                            runner = msg[1]
                            args = msg[2]
                            idx = msg[3]
                            try:
                                result = runner(G, *args)
                                master_write_queue.put(('success', result, idx))
                            except Exception as e:
                                master_write_queue.put(('error', e, idx))
                                # raise
                        elif task == 'run_map_finished':
                            break
                elif task == 'run_collect':
                    collect_once, counter, lock, threshold, args = msg[1:]
                    collected = []
                    try:
                        while True:
                            with lock:
                                if counter.value >= threshold:
                                    break
                            result, inc = collect_once(G, *args)
                            collected.append(result)
                            with lock:
                                counter.value += inc
                                if counter.value >= threshold:
                                    break
                    except Exception as e:
                        worker_write_queue.put(('error', e))
                        continue
                    worker_write_queue.put(('success', collected))
                else:
                    raise NotImplementedError
    except Exception:
        logger.log("Error!")
        import traceback
        for line in traceback.format_exc():
            logger.log(line)
        raise
    finally:
        logger.log("worker exited")


class StatefulPool(object):
    def __init__(self):
        self.n_parallel = 1
        self.workers = []
        self.master_read_queue = None
        self.master_write_queue = None
        self.worker_read_queues = []
        self.worker_write_queues = []
        self.G = SharedGlobal()
        self.G.worker_id = -1

    def terminate(self):
        for worker_read_queue in self.worker_read_queues:
            worker_read_queue.put(('terminate',))
        for worker in self.workers:
            worker.join()
            worker.terminate()
        for queue in (self.worker_read_queues + self.worker_write_queues + [self.master_read_queue,
                                                                            self.master_write_queue]):
            if queue is not None:
                queue.close()
        self.workers = []
        self.worker_read_queues = []
        self.worker_write_queues = []
        self.master_read_queue = None
        self.master_write_queue = None
        self.n_parallel = 1
        self.G = SharedGlobal()
        self.G.worker_id = -1

    def initialize(self, n_parallel):
        if len(self.workers) > 0:  # is not None:
            logger.log("Warning: terminating existing pool")
            self.terminate()
        self.n_parallel = n_parallel
        if self.n_parallel > 1:
            master_read_queue = mp.Queue()
            master_write_queue = mp.Queue()
            for idx in xrange(self.n_parallel):
                worker_read_queue = mp.Queue()
                worker_write_queue = mp.Queue()
                worker = mp.Process(target=worker_main_loop, args=(idx, self.G, worker_read_queue, worker_write_queue,
                                                                   master_read_queue, master_write_queue))
                worker.start()
                self.worker_read_queues.append(worker_read_queue)
                self.worker_write_queues.append(worker_write_queue)
                self.workers.append(worker)
            self.master_read_queue = master_read_queue
            self.master_write_queue = master_write_queue
        atexit.register(self.terminate)

    def run_each(self, runner, args_list=None):
        """
        Run the method on each worker process exactly once, and collect the result of execution.
        The runner method will receive 'G' as its first argument, followed by the arguments
        in the args_list, if any
        :return:
        """
        if args_list is None:
            args_list = [tuple()] * self.n_parallel
        assert len(args_list) == self.n_parallel
        if self.n_parallel > 1:
            results = []
            for worker_read_queue, args in zip(self.worker_read_queues, args_list):
                worker_read_queue.put(('run', runner, args))
            to_raise = None
            for worker_write_queue in self.worker_write_queues:
                status, result = worker_write_queue.get()
                if status == 'success':
                    results.append(result)
                else:
                    to_raise = result
            if to_raise is not None:
                raise to_raise
            return results
        return [runner(self.G, *args_list[0])]

    def run_map(self, runner, args_list):
        """
        Apply the method to each args tupel in the list, and collect the results. There is no guarantee on which
        worker executes which specific task.
        The runner method will receive 'G' as its first argument, followed by the arguments
        in the args_list, if any
        :return:
        """
        if self.n_parallel > 1:
            for idx, args in enumerate(args_list):
                self.master_read_queue.put(('run_map', runner, args, idx))
            # sentinel messages
            for _ in xrange(self.n_parallel):
                self.master_read_queue.put(('run_map_finished',))
            for worker_read_queue in self.worker_read_queues:
                worker_read_queue.put(('run_map',))
            results = [None] * len(args_list)
            to_raise = None
            for _ in xrange(len(args_list)):
                status, result, idx = self.master_write_queue.get()
                if status == 'success':
                    results[idx] = result
                else:
                    to_raise = result
            if to_raise is not None:
                raise to_raise
            return results
        else:
            ret = []
            for args in args_list:
                ret.append(runner(self.G, *args))
            return ret

    def run_collect(self, collect_once, threshold, args=None, show_prog_bar=True):
        """
        Run the collector method using the worker pool. The collect_once method will receive 'G' as
        its first argument, followed by the provided args, if any. The method should return a pair of values.
        The first should be the object to be collected, and the second is the increment to be added.
        This will continue until the total increment reaches or exceeds the given threshold.

        Sample script:

        def collect_once(G):
            return 'a', 1

        stateful_pool.run_collect(collect_once, threshold=3) # => ['a', 'a', 'a']

        :param collector:
        :param threshold:
        :return:
        """
        if args is None:
            args = tuple()
        if self.n_parallel > 1:
            manager = mp.Manager()
            counter = manager.Value('i', 0)
            lock = manager.RLock()
            for worker_read_queue in self.worker_read_queues:
                worker_read_queue.put(('run_collect', collect_once, counter, lock, threshold, args))
            if show_prog_bar:
                pbar = ProgBarCounter(threshold)
            else:
                pbar = None
            last_value = 0
            while True:
                time.sleep(0.1)
                with lock:
                    if counter.value >= threshold:
                        if show_prog_bar:
                            pbar.stop()
                        break
                    if show_prog_bar:
                        pbar.inc(counter.value - last_value)
                    last_value = counter.value
            results = []
            to_raise = None
            for worker_write_queue in self.worker_write_queues:
                status, result = worker_write_queue.get()
                if status == 'success':
                    results.extend(result)
                else:
                    to_raise = result
            if to_raise is not None:
                raise to_raise
            return results
        else:
            count = 0
            results = []
            if show_prog_bar:
                pbar = ProgBarCounter(threshold)
            else:
                pbar = None
            while count < threshold:
                result, inc = collect_once(self.G, *args)
                results.append(result)
                count += inc
                if show_prog_bar:
                    pbar.inc(inc)
            if show_prog_bar:
                pbar.stop()
            return results


singleton_pool = StatefulPool()
