from . import ndict

import numpy as np
import theano
import theano.tensor.shared_randomstreams
import theano.compile
import math, time, sys

# Change recursion limit (for deep theano models)
import sys
sys.setrecursionlimit(10000)

# some config
floatX = theano.config.floatX # @UndefinedVariable
print('[graphy] floatX = '+floatX)

rng = theano.tensor.shared_randomstreams.RandomStreams(0)
rng_curand = rng
if 'gpu' in theano.config.device: # @UndefinedVariable
    import theano.sandbox.cuda.rng_curand
    rng_curand = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(0)

# Shared floating-point Theano variable from a numpy variable
def sharedf(x, target=None, name=None, borrow=False, broadcastable=None):
    if target == None:
        return theano.shared(np.asarray(x, dtype=floatX), name=name, borrow=borrow, broadcastable=broadcastable)
    else:
        return theano.shared(np.asarray(x, dtype=floatX), target=target, name=name, borrow=borrow, broadcastable=broadcastable)

# Shared random normal variable
def sharedrandf(scale, size):
    return sharedf(np.random.normal(0, scale, size=size))

# Construct object from keyword arguments or dictionary
class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)
    def __repr__(self): # nice printing
        return '<%s>' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

# Import rest of the files
from .function import *

from . import misc
from . import misc.data
from . import misc.optim
from . import misc.logger

from . import nodes
from . import graphics
