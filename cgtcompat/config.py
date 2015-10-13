THEANO = 0
CGT = 1
mode = THEANO

def is_theano():
    return mode == THEANO

def is_cgt():
    return mode == CGT
