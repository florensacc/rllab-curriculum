import functools
from collections import defaultdict

import collections
import numpy as np, numpy.random as nr
import numpy as np
import time, sys
import pdb
import itertools, random
import math
import cPickle as pickle
import scipy.optimize
import scipy.misc
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import time
import numpy as np
import itertools
import operator
import scipy
import scipy.optimize
import sys
import lasagne
import lasagne.layers as L
from IPython.display import SVG
from collections import defaultdict, OrderedDict
import argparse
import os
import json

def save_vars(vars, fname):
    cp_data = [var.get_value() for var in vars]
    np.savez_compressed(fname, data=cp_data)


def load_vars(vars, fname, mask=None):
    with Message("Loading params " + fname + " with mask: " + str(mask)):
        if not mask:
            mask = [True for _ in vars]
        loaded = np.load(fname)
        for var, data, use in zip(vars, loaded["data"], mask):
            if use:
                var.set_value(data)

def experiment_iter(max_iter, out_dir, exp_name, checkpoint_freq, checkpoint_vars, start_from=1):
    stats = []
    stats_file = "%s/%s_stats.npz" % (out_dir, exp_name)
    for i in xrange(start_from, max_iter):
        this_stat = OrderedDict()
        yield (i, this_stat)
        if len(this_stat.keys()) != 0:
            print "Experiment: %s\nIteration: %i" % (exp_name, i)
            print tabulate([(k, str(v)) for k,v in this_stat.items()])
            this_stat["iter"] = i
            stats.append(this_stat)
            np.savez_compressed(stats_file, stats=stats)
        if i != 0 and (i % checkpoint_freq) == 0:
            checkpoint_file = "%s/%s_checkpoint_iter_%i.npz" % (out_dir, exp_name, i)
            with Message("Saving " + checkpoint_file):
                save_vars(checkpoint_vars, checkpoint_file)


# used same method as dqn https://github.com/torch/image/blob/24753920d8a91cd2b8944bf60776f68e9f97577a/generic/image.c#L2005
def rgb2y(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return ((0.299 * r) + (0.587 * g) + (0.114 * b))

def wrapped_conv(*args, **kwargs):
    kwargs.pop("image_shape", None)
    kwargs.pop("filter_shape", None)

    return theano.sandbox.cuda.dnn.dnn_conv(*args, **kwargs)

def params_to_cmd_args(params):
    return " ".join(["--%s %s" % p for p in params])


def gen_cmd(base_cmd, exp_base_name, params, runs=3):
    fixed = []
    varying_ks = []
    varying_vss = []
    for k, v in params.items():
        if isinstance(v, collections.Iterable) and not isinstance(v, str):
            varying_ks.append(k)
            varying_vss.append(v)
        else:
            fixed.append((k, v))
    for prod in itertools.product(*varying_vss):
        varied = zip(varying_ks, prod)
        this_params = fixed + varied
        exp_name = "_".join(
            [exp_base_name] +
            ["%s_%s" % p for p in varied])
        for i in xrange(runs):
            run_exp_name = exp_name + ("_run%i" % i)
            cmd_args = params_to_cmd_args(
                [("exp_name", run_exp_name)] + this_params)
            yield (" ".join([base_cmd, cmd_args]), run_exp_name)

# example usage
# base = """docker run \
#     --device /dev/nvidiactl --device /dev/nvidia0 --device /dev/nvidia-uvm \
#     -v ~/.vim:/root/.vim \
#     -v ~/.vimrc:/root/.vimrc \
#     -v /theano_docker:/root/.theano \
#     -v `pwd`/deeprl:/scratch \
#     -v `pwd`/data:/data \
#     neocxi/lab python dqn.py"""
#
# exp_name = "dqn_seaquest"
#
# params = {}
# params["max_iter"] = 1000000
# params["out_dir"] = "/data"
# # params["network_update_freq"] = [50, 500]
# # params["learning_rate"] = [1e-5, 1e-6, 1e-7]
# params["network_update_freq"] = [10000, 20000]
# params["batch_size"] = [16]
#
# import os
# for cmd, name in gen_cmd(base, exp_name, params, runs=2):
#     fname = ("%s.sh" % name)
#     with open(fname, 'w') as f:
#         f.write(cmd)
#     os.system("chmod +x " + fname)
#     os.system("qsub -V -b n -l mem_free=8G,h_vmem=14G -cwd " + fname)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def pack(**kwargs):
    return AttrDict(kwargs)

def pack_dict(dict):
    return AttrDict(dict)

def unpack(x, *keys):
    return tuple(x[k] for k in keys)

def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer

def flatten(lst):
    return sum(lst, [])

def gen_updates(loss, params, args):
    if args.train_params_mask:
        params_to_train = list(select_by(params, args.train_params_mask))
    else:
        params_to_train = params
    if args.optimizer == "adam":
        return lasagne.updates.adam(loss, params_to_train,
                                    learning_rate=args.learning_rate)
    elif args.optimizer == "rmsprop":
        return lasagne.updates.rmsprop(loss, params_to_train,
                                       learning_rate=args.learning_rate, epsilon=0.01, rho=0.95)

def copy_from_params(ps):
    return map(lambda p: np.copy(p.get_value()), ps)

def set_to_params(values, ps):
    for p, v in zip(ps, values):
        p.set_value(np.copy(v))

def multiplicative_perturb(x, min=0.99, max=1.01):
    noise = np.random.rand(*x.shape) * (max - min) + min
    return noise.astype(x.dtype) * x


def copy_vals(vals):
    return [np.copy(v) for v in vals]


def ind_from_vecs(vecs):
    return np.nonzero(vecs)[1].astype('int32')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def init_from_args(params, args):
    if not args.load_params:
        return params
    load_vars(params, args.load_params, args.load_params_mask)
    return params


def select_by(lst1, lst2):
    for a, b in zip(lst1, lst2):
        if b:
            yield a

def new_tensor(name, ndim, dtype):
    import theano.tensor as TT
    return TT.TensorType(dtype, (False,) * ndim)(name)


def new_tensor_like(name, arr_like):
    return new_tensor(name, arr_like.ndim, arr_like.dtype)

def gen_pluck(key):
    def pluck(obj):
        return obj.get(key)
    return pluck

import numpy as np
import scipy.signal
import time

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

MESSAGE_DEPTH = 0
class Message(object):
    def __init__(self, msg, show=True):
        self.msg = msg
        self.show = show
    def __enter__(self):
        if self.show:
            global MESSAGE_DEPTH #pylint: disable=W0603
            print colorize('\t'*MESSAGE_DEPTH + '=: ' + self.msg,'magenta')
            self.tstart = time.time()
            MESSAGE_DEPTH += 1
    def __exit__(self, etype, *args):
        if self.show:
             global MESSAGE_DEPTH #pylint: disable=W0603
             MESSAGE_DEPTH -= 1
             maybe_exc = "" if etype is None else " (with exception)"
             print colorize('\t'*MESSAGE_DEPTH + "done%s in %.3f seconds"%(maybe_exc, time.time() - self.tstart), 'magenta')


# -*- coding: utf-8 -*-

"""Pretty-print tabular data."""


#
# parser = argparse.ArgumentParser(description='Some dqn.')
# parser.add_argument('--rom', default='Seaquest')
# parser.add_argument('--gamma', default=0.99, type=float)
# parser.add_argument('--optimizer', default="adam")
# parser.add_argument('--learning_rate', default=1e-5, type=float)
# parser.add_argument('--max_iter', default=1000000, type=int)
# parser.add_argument('--replay_min', default=50000, type=int)
# parser.add_argument('--network_update_freq', default=10000, type=int)
# parser.add_argument('--evaluation_freq', default=50000, type=int)
# parser.add_argument('--evaluation_len', default=10000, type=int)
# parser.add_argument('--exp_name', default="dqn", type=str)
# parser.add_argument('--out_dir', default="./", type=str)
# parser.add_argument('--checkpoint_freq', default=100000, type=int)
# parser.add_argument('--batch_size', default=32, type=int)
# parser.add_argument('--network', default=None, type=str)
# parser.add_argument('--network_args', default=None, type=str)
# parser.add_argument('--min_eps', default=0.01, type=float)
# parser.add_argument('--algo', default=None, type=str)
# parser.add_argument('--terminate_per_life', default=False, type=str2bool)
# parser.add_argument('--temporal_frames', default=3, type=int)
# parser.add_argument('--load_params', default=None, type=str)
# parser.add_argument('--load_params_mask', default=None, type=json.loads)
# parser.add_argument('--train_params_mask', default=None, type=json.loads)
# parser.add_argument('--lbfgs_iters', default=10, type=int)
# parser.add_argument('--penalty_type', default="action", type=str)
# parser.add_argument('--test_batch_size', default=50, type=int)
# parser.add_argument('--dup_factor', default=1, type=int)
# parser.add_argument('--validation_batch_size', default=1000, type=int)
# parser.add_argument('--shuffle_update_order', default=False, type=str2bool)
# parser.add_argument('--penalty_rate', default=1., type=float)
# parser.add_argument('--eval_runs', default=5, type=int)
# parser.add_argument('--train_runs', default=1, type=int)
# parser.add_argument('--train_switch_freq', default=0, type=int)
# parser.add_argument('--n_processes', default=2, type=int)
# parser.add_argument('--red_factor', default=15, type=int)
# parser.add_argument('--update_slave', default=True, type=str2bool)
# parser.add_argument('--dropout_max', default=False, type=str2bool)
# parser.add_argument('--start_from', default=1, type=int)
# parser.add_argument('--memory_size', default=190000, type=int)
# parser.add_argument('--no_replay', default=False, type=str2bool)
# parser.add_argument('--dropout_rollout', default=False, type=str2bool)
# parser.add_argument('--load_expert_params', default=None, type=str)
# parser.add_argument('--regress_q_pi', default=False, type=str2bool)
# parser.add_argument('--select_max_by_current_q', default=False, type=str2bool)
