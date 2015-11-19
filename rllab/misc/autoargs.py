from rllab.misc.console import colorize

def arg(name, type, help, mapper=None):
    def wrap(fn):
        assert fn.__name__ == '__init__'
        if not hasattr(fn, '_autoargs_info'):
            fn._autoargs_info = dict()
        fn._autoargs_info[name] = dict(type=type, help=help, mapper=mapper)
        return fn
    return wrap


def prefix(prefix):
    def wrap(fn):
        assert fn.__name__ == '__init__'
        fn._autoargs_prefix = prefix
        return fn
    return wrap


def _get_prefix(cls):
    from rllab.mdp.base import MDP
    from rllab.policy.base import Policy
    from rllab.vf.base import ValueFunction
    from rllab.algo.base import Algorithm


    if hasattr(cls.__init__, '_autoargs_prefix'):
        return cls.__init__._autoargs_prefix
    elif issubclass(cls, MDP):
        return 'mdp_'
    elif issubclass(cls, Algorithm):
        return 'algo_'
    elif issubclass(cls, ValueFunction):
        return 'vf_'
    elif issubclass(cls, Policy):
        return 'policy_'
    else:
        return ""

def _get_info(cls):
    if hasattr(cls.__init__, '_autoargs_info'):
        return cls.__init__._autoargs_info
    return {}


def add_args(_):
    def _add_args(cls, parser):
        args_info = _get_info(cls)
        prefix = _get_prefix(cls)
        for arg_name, arg_info in args_info.iteritems():
            parser.add_argument('--' + prefix + arg_name, help=arg_info['help'], type=arg_info['type'])
    return _add_args


def new_from_args(_):
    def _new_from_args(cls, parsed_args, *args):
        args_info = _get_info(cls)
        prefix = _get_prefix(cls)
        params = dict()
        for arg_name, arg_info in args_info.iteritems():
            prefixed_arg_name = prefix + arg_name
            if hasattr(parsed_args, prefixed_arg_name) and getattr(parsed_args, prefixed_arg_name):
                if arg_info['mapper']:
                    params[arg_name] = arg_info['mapper'](getattr(parsed_args, prefixed_arg_name))
                else:
                    params[arg_name] = getattr(parsed_args, prefixed_arg_name)
                print colorize("using argument %s with value %s" % (arg_name, params[arg_name]), "yellow")
        return cls(*args, **params)
    return _new_from_args
