
import numpy as np
import theano
import theano.tensor as TT
import multiprocessing as mp
from contextlib import closing


def new_shared_mem_array(init_val):
    typecode = init_val.dtype.char
    arr = mp.Array(typecode, np.prod(init_val.shape))
    nparr = np.frombuffer(arr.get_obj())
    nparr.shape = init_val.shape
    return nparr


def init_worker(x):
    global x_
    x_ = x
    x_var = theano.shared(np.zeros_like(x), borrow=True)
    x_var.set_value(x_, borrow=True)
    global f_update
    f_update = theano.function(
        inputs=[],
        outputs=[],
        updates=[(x_var, x_var + TT.ones_like(x_var))]
    )


def update_x(*args, **kwargs):
    global f_update
    f_update()


def main():
    x = np.zeros(10)
    x = new_shared_mem_array(x)
    with closing(mp.Pool(initializer=init_worker, initargs=(x,))) as p:
        p.map(update_x, range(10000))
    print(x)

main()
