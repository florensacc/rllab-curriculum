def merge_dict(x, y):
    z = x.copy()
    z.update(y)
    return z

def extract(x, *keys):
    return tuple(x[k] for k in keys)
