
"""
Legacy implementation of barrier object for multiprocessing, which did not have
a barrier in Python2.
"""

from multiprocessing import Semaphore

class Barrier:

    def __init__(self, n):
        self.n = n  # number of threads subject to barrier
        self.count = 0  # number of threads at the barrier
        self.mutexIn = Semaphore(1)  # initialize as 'availalble'
        self.mutexOut = Semaphore(1)
        self.barrier = Semaphore(0)  # initialize as 'blocked'

    def wait(self):
        self.mutexIn.acquire()  # one at a time
        self.count += 1  # check in

        if self.count < self.n:  # if not the last one
            self.mutexIn.release()  # let others check in
        else:  # if the last one
            self.barrier.release()  # begin barrier release chain
                                    # mutexIn stays acquired, no re-entry yet

        self.barrier.acquire()  # wait until all have checked in
        self.barrier.release()  # all threads pass simultaneously

        self.mutexOut.acquire()  # one at a time
        self.count -= 1  # checkout
        if self.count == 0:  # if the last one
            self.barrier.acquire()  # block the barrier
            self.mutexIn.release()  # allow re-entry
        self.mutexOut.release()  # allow next checkout
