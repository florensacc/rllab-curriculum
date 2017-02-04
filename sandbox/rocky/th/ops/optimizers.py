import torch.optim


class DelayedOptimizer(object):
    def __init__(self, cls, kwargs):
        self.cls = cls
        self.kwargs = kwargs

    def bind(self, params):
        return self.cls(params, **self.kwargs)


def get_optimizer(name, **kwargs):
    if name == 'adam':
        return DelayedOptimizer(torch.optim.Adam, kwargs)
    elif name == 'adamax':
        return DelayedOptimizer(torch.optim.Adamax, kwargs)
    elif name == 'rmsprop':
        return DelayedOptimizer(torch.optim.RMSprop, kwargs)
    else:
        raise NotImplementedError
