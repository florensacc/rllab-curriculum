import numpy as np
import scipy

def weighted_sample(weights, objects):
    """Return a random item from objects, with the weighting defined by weights 
    (which must sum to 1)."""
    cs = np.cumsum(weights) #An array of the weights, cumulatively summed.
    idx = sum(cs < np.random.rand()) #Find the index of the first weight over a random value.
    return objects[idx]

# compute softmax for each row
def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    expx = np.exp(shifted)
    return expx / np.sum(expx, axis=1, keepdims=True)

# compute entropy for each row
def cat_entropy(x):
    return -np.sum(x * np.log(x), axis=1)

# compute perplexity for each row
def cat_perplexity(x):
    return np.exp(cat_entropy(x))

def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]
