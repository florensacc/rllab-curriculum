import zmq
import zlib, pickle
import os
from policy import DiscreteNNPolicy
from algo.utrpo import UTRPO
from mdp.base import MDP
from mdp.atari_mdp import AtariMDP, OBS_RAM
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import inspect
import os
import numpy as np
import scipy.optimize
import lasagne.layers as L
import operator
import sys
from misc.console import Message, log, prefix_log
from misc.tensor_utils import flatten_tensors, unflatten_tensors
from collections import defaultdict
import multiprocessing
from joblib.pool import MemmapingPool
from joblib.parallel import SafeFunction
import subprocess
import theano
import theano.tensor as T
import cloudpickle
import zmq


def launch_sampler(gen_sampler):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:12577")
    print 'waiting for message...'
    message = socket.recv()
    socket.send('ack')
    with gen_sampler(message) as sampler:
        message = pickle.loads(socket.recv())
        ret = sampler.collect_samples(*message)
        socket.send(cloudpickle.dumps(ret))
