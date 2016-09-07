import zerorpc, pickle

class ParAlgMaster(object):
    """
    Object living on master computer, used for running a parallel algorithm
    """
    def __init__(self, namespace):
        addrs = self.get_slave_addrs(namespace)
        self.clients = []
        for addr in addrs:
            client = zerorpc.Client()
            client.connect(addr)
            self.clients.append(client)
        self.name2reducer = {}
    def train(self):
        _sync([client("train_for", 1, async=True) for client in self.clients])
        all_params = _sync([client("get_params", async=True) for client in self.clients])
        avg_params = reduce_params(all_params, self.name2reducer)
        _sync([client("set_params", avg_params, async=True) for client in self.clients])
    def set_reducer(self, name, reducer):
        self.name2reducer[name] = reducer
    def get_slave_addrs(self, ns):
        raise NotImplementedError

def _maybe_unpickle(x):
    if isinstance(x, str) and x.startswith('\x80'):
        return pickle.loads(x)
    else:
        return x

def _sync(futures):
    return [_maybe_unpickle(f.get()) for f in futures]

def reduce_params(name2values, name2reducer):
    out = {}
    nvalues = len(name2values)
    for name in list(name2reducer.keys()):
        reducer = name2reducer.get(name, "average")
        if reducer == "average":
            out[name] = name2values[0].copy()
            for name2value in name2values[1:]:
                out[name] += name2value[name]
            out[name] /= nvalues
        elif reducer == "add":
            out[name] = name2values[0].copy()
            for name2value in name2values[1:]:
                out[name] += name2value[name]
        else:
            raise NotImplementedError
    return out