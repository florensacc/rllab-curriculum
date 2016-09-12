from chainer import links as L
import numpy as np


def init_like_torch(link):
    # Mimic torch's default parameter initialization
    # Each pre-activation unit has variance 1
    # TODO(muupan): Use chainer's initializers when it is merged
    for l in link.links():
        if isinstance(l, L.Linear):
            out_channels, in_channels = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape).astype(np.float32)
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape).astype(np.float32)
        elif isinstance(l, L.Convolution2D):
            out_channels, in_channels, kh, kw = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels * kh * kw)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape).astype(np.float32)
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape).astype(np.float32)

