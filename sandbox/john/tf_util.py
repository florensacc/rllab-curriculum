from collections import namedtuple
import numpy as np
import tensorflow as tf # pylint: ignore-module

# ================================================================
# Import all names into common namespace
# ================================================================

from tensorflow import (
    tanh, sigmoid, exp, greater, log, square, add_n, add, sqrt, abs, maximum, minimum,    
    equal, not_equal,
    reshape,
    matmul, stop_gradient, tile, constant, pack, assign,
    random_normal, random_uniform,
)
nn = tf.nn
relu = nn.relu
softmax = nn.softmax
softplus = nn.softplus
conv2d = nn.conv2d
conv2d_transpose = nn.conv2d_transpose
xent_sigmoid = tf.nn.sigmoid_cross_entropy_with_logits
xent_softmax = tf.nn.softmax_cross_entropy_with_logits
floatX = "float32"
tf_floatX = tf.float32
AdamOptimizer = tf.train.AdamOptimizer
GradientDescentOptimizer = tf.train.GradientDescentOptimizer
clip = tf.clip_by_value

# Make consistent with numpy 
# ----------------------------------------

def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def std(x, axis=None, keepdims=False):
    return var(x, axis=axis, keepdims=keepdims)
def max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=True)
    return mean(square(x - meanx), axis=axis, keepdims=keepdims)
def concatenate(arrs, axis=0):
    return tf.concat(axis, arrs)
def argmax(x, axis=None):
    return tf.argmax(x, dimension=axis)


# Extras 
# ----------------------------------------
def l2loss(params):
    if len(params) == 0:
        return constant(0.0)
    else:
        return add_n([sum(square(p)) for p in params])
def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
def toFloatX(x):
    if isinstance(x, tf.Tensor):
        return tf.cast(x, tf_floatX)
    else:
        return x.astype(floatX)

# ================================================================
# Global session 
# ================================================================

SESSION = tf.Session()

ALREADY_INITIALIZED = set()
def initialize():
    global ALREADY_INITIALIZED
    new_variables = set(tf.all_variables()) - ALREADY_INITIALIZED
    SESSION.run(tf.initialize_variables(new_variables))
    ALREADY_INITIALIZED.update(new_variables)
    

def eval(expr, feed_dict=None):
    if feed_dict is None: feed_dict = {}
    return SESSION.run(expr, feed_dict=feed_dict)

# ================================================================
# Modules 
# ================================================================
# Modules let you write a function corresponding to a parameterized
# function. New variables will be created the first time you call it,
# and retrieved on subsequent calls.


class module(object):
    current = None
    def __init__(self, f):
        self.firstcall = True
        self.varidx = None
        self.vars = []
        self.f = f
    def __call__(self, *inputs, **kwargs):
        prev = module.current
        module.current = self
        self.varidx = 0
        outputs = self.f(*inputs, **kwargs)
        self.firstcall = False
        module.current = prev
        return outputs
    def get_variable(self, *args, **kws):
        if self.firstcall:
            v = tf.Variable(*args, **kws)
            self.vars.append(v)
            return v
        else:
            v = self.vars[self.varidx]
            self.varidx += 1
            return v

# ================================================================
# Initializations
# ================================================================

IIDGaussian = namedtuple("IIDGaussian", ["mean", "std"])
IIDGaussian.__new__.__defaults__ = (0, 1)
IIDUniform = namedtuple("IIDUniform", ["low", "high"])
NormalizedColumns = namedtuple("NormalizedColumns", "std")
Constant = namedtuple("Constant", ["constant"])

def init_array(init, shape):
    if isinstance(init, IIDGaussian):
        return np.random.normal(size=shape, loc=init.mean, scale=init.std).astype(floatX)
    elif isinstance(init, IIDUniform):
        return np.random.uniform(size=shape, low=init.low, high=init.high)
    elif isinstance(init, Constant):
        return init.constant*np.ones(shape, floatX)
    elif isinstance(init, NormalizedColumns):
        out = np.random.randn(*shape).astype(floatX)
        out *= init.std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return out
    else:
        raise ValueError("Invalid initializer %s"%init)

# ================================================================
# Costs 
# ================================================================

def gaussian_bits(mean, logstd):
    """
    Computes KL divergence elementwise between N(mean, exp(logstd)) and N(0, 1)
    """
    logvar = 2 * logstd
    return 0.5 * (tf.square(mean) + tf.exp(logvar) - logvar - 1.0)

# ================================================================
# Basic Stuff 
# ================================================================

def Input(shape, dtype=floatX, name=None):
    return tf.placeholder(dtype, shape=shape, name=name)

def Variable(value=None, shape=None, init=None, name=None):
    if value is not None:
        initial_value = value
    elif (shape is not None) and (init is not None):
        initial_value = init_array(init, shape)
    else:
        raise ValueError
    if module.current is None:
        v = tf.Variable(initial_value=initial_value, dtype=tf_floatX, name=name)
    else:
        v = module.current.get_variable(initial_value=initial_value, dtype=tf_floatX, name=name)
    VARIABLES[v._variable] = v
    return v

def function(inputs, outputs, updates=None):
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates)
    elif isinstance(outputs, dict):
        f = _Function(inputs, list(outputs.values()), updates)
        return lambda *inputs : dict(list(zip(list(outputs.keys()), f(*inputs))))
    else:
        f = _Function(inputs, [outputs], updates)
        return lambda *inputs : f(*inputs)[0]

class _Function(object):
    def __init__(self, inputs, outputs, updates):
        self.inputs = inputs
        self.outputs = list(outputs)
        self.updates = [] if updates is None else updates
    def __call__(self, *inputvals):
        assert len(inputvals) == len(self.inputs)
        feed_dict = dict(list(zip(self.inputs, inputvals)))
        results = SESSION.run(self.outputs + self.updates, feed_dict=feed_dict)
        if any(r is not None and np.isnan(r).any() for r in results):
            raise RuntimeError("Nan detected")
        return results[:len(self.outputs)]

# ================================================================
# Bigger blocks 
# ================================================================

def dense(x, n_out, n_in=None, weight_init=NormalizedColumns(1.0), bias_init=Constant(0.0)):
    if n_in is None: 
        n_in = x.get_shape()[1]
        assert n_in is not None
    weight = Variable(init=weight_init, shape=(n_in, n_out))
    bias = Variable(init=bias_init, shape=(1, n_out))
    return matmul(x, weight) + bias


def densenobias(x, n_out, n_in=None, weight_init=NormalizedColumns(1.0)):
    if n_in is None: 
        n_in = x.get_shape()[1]
        assert n_in is not None
    weight = Variable(init=weight_init, shape=(n_in, n_out))
    return matmul(x, weight)

def batchnorm(x, post=False, eps=1e-8):
    ndim = len(x.get_shape())
    if ndim == 2:
        demeanx = x - mean(x, axis=0, keepdims=True)
        ms = mean(square(demeanx), axis=0, keepdims=True)
        rms = sqrt(ms + eps)
        z = demeanx / rms
    elif ndim == 4:
        demeanx = x - tf.reduce_mean(x, reduction_indices=[0,1,2], keep_dims=True)
        ms = tf.reduce_mean(square(demeanx), reduction_indices=[0,1,2], keep_dims=True)
        rms = sqrt(ms + eps)
        z = demeanx / rms        
    else:
        raise NotImplementedError
    if post:
        raise NotImplementedError
        # x_size = x.get_shape()[1]
        # postmul = Variable(np.ones((1, x_size), floatX))
        # postadd = Variable(np.zeros((1, x_size), floatX))
        # return z * postmul + postadd
    else:
        return z

# ================================================================
# Graph traversal
# ================================================================

VARIABLES = {}

def get_variables(outputs):
    if not isinstance(outputs, (list,tuple)): outputs = [outputs]
    return [VARIABLES[v] for v in topsorted(outputs) if v in VARIABLES]

def get_parents(node):
    return node.op.inputs

def topsorted(outputs):
    """
    Topological sort via non-recursive depth-first search
    """
    assert isinstance(outputs, (list,tuple))
    marks = {}
    out = []
    stack = [] #pylint: disable=W0621
    # i: node
    # jidx = number of children visited so far from that node
    # marks: state of each node, which is one of
    #   0: haven't visited
    #   1: have visited, but not done visiting children
    #   2: done visiting children
    for x in outputs:
        stack.append((x,0))
        while stack:
            (i,jidx) = stack.pop()
            if jidx == 0:
                m = marks.get(i,0)
                if m == 0:
                    marks[i] = 1
                elif m == 1:
                    raise ValueError("not a dag")
                else:
                    continue
            ps = get_parents(i)
            if jidx == len(ps):
                marks[i] = 2
                out.append(i)
            else:
                stack.append((i,jidx+1))
                j = ps[jidx]
                stack.append((j,0))
    return out
        

# ================================================================
# Flat vectors 
# ================================================================

def var_shape(x):
    out = [k.value for k in x.get_shape()]    
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [numel(v)])
        for (v, grad) in zip(var_list, grads)])

class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf_floatX):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([np.prod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype,[total_size])
        start=0
        assigns = [] 
        for (shape,v) in zip(shapes,var_list):
            size = np.prod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start+size],shape)))
            start+=size
        self.op = tf.group(*assigns)
    def __call__(self, theta):
        SESSION.run(self.op, feed_dict={self.theta:theta})

class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])
    def __call__(self):
        return SESSION.run(self.op)

# ================================================================
# Misc 
# ================================================================


def fancy_slice_2d(X, inds0, inds1):
    """
    like numpy X[inds0, inds1]
    XXX this implementation is bad
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)