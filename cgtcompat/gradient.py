from .config import is_theano, is_cgt#mode, THEANO, CGT
import theano
import theano.tensor as T

def grad(a, b):
    if is_theano():
        return theano.gradient.grad(a, b)
