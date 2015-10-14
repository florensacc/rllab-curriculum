import gradient
from .config import is_theano, is_cgt
import theano

def function(*args, **kwargs):
    if is_theano():
        return theano.function(*args, **kwargs)


def shared(*args, **kwargs):
    if is_theano():
        return theano.shared(*args, **kwargs)
