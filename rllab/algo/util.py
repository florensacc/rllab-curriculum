def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **compact(kwargs))
    else:
        raise NotImplementedError



