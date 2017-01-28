import numpy as np


def sample_matrix_row(M, size, replace=False):
    if size > M.shape[0] or replace:
        indices = np.random.randint(0, M.shape[0], size)
    else:
        indices = np.random.choice(M.shape[0], size)
    return M[indices, :]
