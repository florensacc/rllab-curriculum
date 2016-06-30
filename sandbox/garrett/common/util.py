import numpy as np
import theano
import theano.tensor as TT

def init_weights(*shape):
    return theano.shared((np.random.randn(*shape) * 0.01).astype(theano.config.floatX))
