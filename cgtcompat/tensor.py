from .config import is_theano, is_cgt#mode, THEANO, CGT
import theano
import theano.tensor as T

def matrix(name):
    if is_theano():
        return T.matrix(name)

def vector(name):
    if is_theano():
        return T.vector(name)

def scalar(name):
    if is_theano():
        return T.scalar(name)

def mean(a):
    if is_theano():
        return T.mean(a)

def tile(*args, **kwargs):#a, b):
    if is_theano():
        return T.tile(*args, **kwargs)#a, b)

def square(a):
    if is_theano():
        return T.square(a)

def log(a):
    if is_theano():
        return T.log(a)

def exp(a):
    if is_theano():
        return T.exp(a)

def prod(*args, **kwargs):
    if is_theano():
        return T.prod(*args, **kwargs)
