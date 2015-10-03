from misc.tensor_utils import cg
import numpy as np
import numpy.random as nr

A = nr.rand(3,3)
A = np.dot(A.T, A)

def prod(x):
    return np.dot(A, x)

b = nr.rand(3)

x = cg(prod, b, np.zeros(3))
true_x = np.dot(np.linalg.inv(A), b)

print x
print true_x
