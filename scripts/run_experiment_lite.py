from __future__ import print_function
import sys
sys.path.append(".")

from rllab.misc.ext import is_iterable, set_seed
from rllab.misc.console import StubClass, StubMethodCall
from rllab import config
import rllab.misc.logger as logger
import argparse
import os.path as osp
import datetime
import dateutil.tz
import ast
import uuid
import cPickle as pickle
import inspect


def infer_name(obj, prefix):
    if prefix and len(prefix) > 0:
        return prefix
    return "algo"


def prepend(a, b):
    if a and len(a) > 0:
        return a + "." + b
    return b


def concretize(maybe_stub, cls_set, prefix=""):
    if isinstance(maybe_stub, StubMethodCall):
        obj = concretize(maybe_stub.obj, cls_set)
        method = getattr(obj, maybe_stub.method_name)
        arg_names = inspect.getargspec(method).args[1:]
        args = [concretize(x, cls_set, prepend(prefix, name)) for x, name in zip(maybe_stub.args, arg_names)]
        kwargs = dict([(k, concretize(v, cls_set, prepend(prefix, k))) for k, v in maybe_stub.kwargs.iteritems()])
        return lambda: method(*args, **kwargs)
    elif isinstance(maybe_stub, StubClass):
        if not hasattr(maybe_stub, "__stub_cache"):
            arg_names = inspect.getargspec(maybe_stub.proxy_class.__init__).args[1:]
            args = [concretize(x, cls_set, prepend(prefix, name)) for x, name in zip(maybe_stub.args, arg_names)]
            kwargs = dict([(k, concretize(v, cls_set, prepend(prefix, k))) for k, v in maybe_stub.kwargs.iteritems()])
            maybe_stub.__stub_cache = maybe_stub.proxy_class(*args, **kwargs)
        cls_set.add((
            maybe_stub.__stub_cache,
            infer_name(maybe_stub.__stub_cache, prefix)
        ))
        return maybe_stub.__stub_cache
    else:
        return maybe_stub


def run_experiment(argv):

    default_log_dir = "{PROJECT_PATH}/data"
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize_mdp', type=ast.literal_eval,
                        default=False,
                        help="Whether to normalize the mdp's actions to take "
                             "value between -1 and 1")
    parser.add_argument('--random_mdp', type=ast.literal_eval,
                        default=False,
                        help="Whether to reinit the mdp from random physical "
                             "model every episode")
    parser.add_argument('--action_delay', type=int,
                        default=0,
                        help="Time steps delayed injected into MDP")
    parser.add_argument('--obs_noise', type=float,
                        default=0,
                        help="Guassian noise added to obs")
    parser.add_argument('--n_parallel', type=int, default=1,
                        help='Number of parallel workers to perform rollouts.')
    parser.add_argument('--exp_name', type=str, default=default_exp_name,
                        help='Name of the experiment.')
    parser.add_argument('--log_dir', type=str, default=default_log_dir,
                        help='Path to save the log and iteration snapshot.')
    parser.add_argument('--snapshot_mode', type=str, default='all',
                        help='Mode to save the snapshot. Can be either "all" '
                             '(all iterations will be saved), "last" (only '
                             'the last iteration will be saved), or "none" '
                             '(do not save snapshots)')
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--params_log_file', type=str, default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--plot', type=ast.literal_eval, default=False,
                        help='Whether to plot the iteration results')
    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')
    parser.add_argument('--args_data', type=str,
                        help='Pickled data for stub objects')

    args = parser.parse_args(argv[1:])


    from rllab.sampler import parallel_sampler
    parallel_sampler.config_parallel_sampler(args.n_parallel, args.seed)
    if args.plot:
        from rllab.plotter import plotter
        plotter.init_worker()

    # read from stdin
    data = pickle.loads(args.args_data)

    if args.seed is not None:
        set_seed(args.seed)

    log_dir = args.log_dir.format(PROJECT_PATH=config.PROJECT_PATH)
    exp_dir = osp.join(log_dir, args.exp_name)
    tabular_log_file = osp.join(exp_dir, args.tabular_log_file)
    text_log_file = osp.join(exp_dir, args.text_log_file)
    params_log_file = osp.join(exp_dir, args.params_log_file)

    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    logger.set_snapshot_dir(exp_dir)
    logger.set_snapshot_mode(args.snapshot_mode)
    logger.push_prefix("[%s] " % args.exp_name)

    cls_set = set()
    f_call = concretize(data, cls_set)

    top_cls = {}
    for cls, name in sorted(cls_set, key=lambda x: len(x[1].split('.'))):
        if cls not in top_cls.values():
            top_cls[name] = cls

    logger.log_parameters_lite(params_log_file, args, data)

    maybe_iter = f_call()
    if is_iterable(maybe_iter):
        for _ in maybe_iter:
            pass

    logger.set_snapshot_mode(prev_mode)
    logger.set_snapshot_dir(prev_snapshot_dir)
    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)
    logger.pop_prefix()

if __name__ == "__main__":
    run_experiment(sys.argv)
