
import numpy as np


def cg(f_Ax, b, par_objs, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312

    par_objs: must be a struct containing the following

    'z': must refer to the shared variables to which f_Ax writes.
    'p': shared variable used internally here.
    'x': the shared variable to which the output is written.
    'brk': shared boolean
    'barrier': one multiprocessing barrier (for n_parallel)

    z, p, and x must all be the same size as the input b.

    (Could run regular serial version, since f_Ax will be parallelized, but
    parallelizing here reduces overall memory usage.)

    """

    z = par_objs.z
    p = par_objs.p
    x = par_objs.x
    par_objs.brk.value = False

    p[:] = b
    # p = b.copy()
    par_objs.barrier.wait()
    r = b.copy()
    # x = np.zeros_like(b)
    x.fill(0.)
    rdotr = r.dot(r)

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
        f_Ax(p)  # (parallelized, writes to z)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        # p = r + mu * p
        p[:] = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            par_objs.brk.value = True
        par_objs.barrier.wait()
        if par_objs.brk.value:
            break

    if callback is not None:
        callback(x)
    if verbose:
        print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x  # (or access this elsewhere)


def cg_worker(f_Ax, par_objs, cg_iters=10):
    par_objs.barrier.wait()
    for i in range(cg_iters):
        f_Ax(par_objs.p)  # (parallel)
        par_objs.barrier.wait()
        if par_objs.brk.value:
            break
