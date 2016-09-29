
import numpy as np


def cg(f_Ax, b, par_objs, rank, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312

    Parallelized.

    Could run regular serial version, since f_Ax will be parallelized, but
    parallelizing here reduces overall memory usage.

    par_objs: must be a dictionary containing the following

    'z': must refer to the shared variables to which f_Ax writes.
    'p': shared variable used internally only.
    'x': the shared variable to which the output is written.
    'brk': shared int
    'barrier': one multiprocessing barrier (for n_parallel)

    z, p, and x must all be the same size as the input b.
    """
    if rank == 0:
        _cg_master(f_Ax, b, par_objs, cg_iters, callback, verbose, residual_tol)
    else:
        _cg(f_Ax, par_objs, cg_iters)


def _cg_master(f_Ax, b, par_objs, cg_iters, callback, verbose, residual_tol):
    z = par_objs['z']
    p = par_objs['p']
    x = par_objs['x']
    brk = par_objs['brk']
    barrier = par_objs['barrier']

    p[:] = b
    barrier.wait()
    r = b.copy()
    x.fill(0.)
    rdotr = r.dot(r)
    brk.value = 0

    if verbose:
        fmtstr = "%10i %10.3g %10.3g"
        titlestr = "%10s %10s %10s"
        print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        # z = f_Ax(p)
        f_Ax(p)  # (parallel, writes to persistent shared variable, z)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        # p = r + mu * p
        p[:] = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            brk.value = 1
        barrier.wait()
        if brk.value > 0:
            break

    if callback is not None:
        callback(x)
    if verbose:
        print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631


def _cg(f_Ax, par_objs, cg_iters):
    p = par_objs['p']
    brk = par_objs['brk']
    barrier = par_objs['barrier']

    barrier.wait()
    for i in range(cg_iters):
        f_Ax(p)  # (parallel)
        barrier.wait()
        if brk.value > 0:
            break
