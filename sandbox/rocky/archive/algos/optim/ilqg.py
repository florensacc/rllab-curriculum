# Source: Control-Limited Differential Dynamic Programming by Y Tassa et al.

import numpy as np
import scipy as sp
from numpy.linalg import LinAlgError
from rllab.misc.ext import extract
from rllab.algos.optim.common import forward_pass, convexify

__all__ = [
        "solve"
]
#class NonPositiveDefiniteError(Exception):
#    pass

def backward_pass(x, u, f_forward, f_cost, f_final_cost, grad_hints, reg):
    Dx = x.shape[1]
    N, Du = u.shape
    print('convexifying...')
    fx, fu, cx, cu, cxx, cxu, cuu = extract(
        convexify(x, u, f_forward, f_cost, f_final_cost),
        "fx", "fu", "cx", "cu", "cxx", "cxu", "cuu"
    )
    print('convexified')

    Vx = np.zeros((N+1, Dx))
    Vxx = np.zeros((N+1, Dx, Dx))
    Vx[N] = cx[N]
    Vxx[N] = cxx[N]
    k = np.zeros((N, Du))
    K = np.zeros((N, Du, Dx))
    Quu = np.zeros((N, Du, Du))
    
    # dVx and dVxx are used for computing the expected reduction in cost
    # dV ~ -alpha*Q_u.T*k - alpha^2*k.T*Q_uu*k
    dVx = 0
    dVxx = 0
    
    for t in range(N-1, -1, -1):
        Qx = cx[t] + fx[t].T.dot(Vx[t+1])
        Qu = cu[t] + fu[t].T.dot(Vx[t+1])
        Qxx = cxx[t] + fx[t].T.dot(Vxx[t+1]).dot(fx[t])
        Qxx = 0.5*(Qxx+Qxx.T)
        # As proposed in Fast Model Predictive Control for Reactive Robotic Swimming by Yuval Tassa et al.
        #reg_Vxxt = Vxx[t+1] + reg*np.eye(Vxx[t+1].shape[0])
        Qux = cxu[t].T + fu[t].T.dot(Vxx[t+1]).dot(fx[t])
        Quu[t] = cuu[t] + fu[t].T.dot(Vxx[t+1]).dot(fu[t]) + reg*np.eye(Quu[t].shape[0])
        try:
            U_Quu = np.linalg.cholesky(Quu[t])
            L_Quu = U_Quu.T
            k[t] = -sp.linalg.solve_triangular(U_Quu, sp.linalg.solve_triangular(L_Quu, Qu, lower=True))
            K[t] = -sp.linalg.solve_triangular(U_Quu, sp.linalg.solve_triangular(L_Quu, Qux, lower=True))
        except LinAlgError as e:#Exception as e:
            print('error in %d' % t)
            raise 
        
        Vx[t] = Qx + Qux.T.dot(k[t])
        Vxx[t] = Qxx + Qux.T.dot(K[t])

        if np.any(abs(Vxx) > 1e10):
            raise LinAlgError('failed')
        
        dVx += np.inner(k[t], Qu)
        dVxx += 0.5*k[t].T.dot(Quu[t]).dot(k[t])
    return dict(Vx=Vx, Vxx=Vxx, k=k, K=K, dVx=dVx, dVxx=dVxx, Quu=Quu)

def solve(
        x0,
        uinit,
        f_forward,
        f_cost,
        f_final_cost,
        grad_hints=None,
        max_iter=100,
        min_reduction=0,
        max_line_search_iter=8,
        lambda_init=1.0,
        lambda_scale_factor=10.0,
        lambda_max=1e6,
        lambda_min=1e-3,
        rel_tol=1e-3,
        abs_tol=1e-4,
        ):
    
    Dx = len(x0)
    Du = uinit.shape[1]
    N = len(uinit)
    
    u = uinit

    lambda_ = lambda_init
    
    for itr in range(max_iter):
        x, cost = extract(
                forward_pass(x0, u, f_forward, f_cost, f_final_cost),
                "x", "cost"
        )

        bwd_succeeded = False
        while not bwd_succeeded:
            try:
                print('backward pass...')
                Vx, Vxx, k, K, dVx, dVxx, Quu = extract(
                    backward_pass(x, u, f_forward, f_cost, f_final_cost, grad_hints, reg=lambda_),
                    "Vx", "Vxx", "k", "K", "dVx", "dVxx", "Quu"
                )
                bwd_succeeded = True
            except LinAlgError as e:#NonPositiveDefiniteError:
                #print e
                lambda_ = max(lambda_ * lambda_scale_factor, lambda_min)
                #print 'increasing lambda to %f' % lambda_
                #import ipdb; ipdb.set_trace()
                if lambda_ > lambda_max:
                    break

        if not bwd_succeeded:
            raise ValueError("Cannot find positive definition solution even with very large regularization")

        alpha = 1
        fwd_succeeded = False
        for _ in range(max_line_search_iter):
            #print 'alpha: ', alpha
            # unew is different from u+alpha*k because of the feedback control term
            xnew, cnew, unew = extract(
                forward_pass(x0, u+alpha*k, f_forward, f_cost, f_final_cost, K=K, xref=x),
                "x", "cost", "u"
            )
            #print 'cnew:', cnew#u+alpha*k
            dcost = cost - cnew
            expected = -alpha*(dVx+alpha*dVxx)
            if expected < 0:
                raise ValueError('should not happen')
            if dcost / expected > min_reduction:
                fwd_succeeded = True
                break
            else:
                alpha = alpha * 0.5
                print('expected decrease is less than min reduction: decreasing alpha to %f' % alpha)
        if fwd_succeeded:
            lambda_ = min(lambda_, 1) / lambda_scale_factor#, lambda_min)
            if lambda_ < lambda_min:
                lambda_ = lambda_min#0
            u = unew
            if abs(cost - cnew) / abs(cost) < rel_tol or abs(cost - cnew) < abs_tol:
                print('no cost improvement')
                break
            cost = cnew
        else:
            lambda_ = max(lambda_, 1) * lambda_scale_factor
            if lambda_ > lambda_max:
                print("Cannot improve objective even with large regularization")
                break
        yield dict(u=u, K=K, k=k, x=x, Quu=Quu, cost=cost)
        print('lambda:', lambda_)
    yield dict(u=u, K=K, k=k, x=x, Quu=Quu, cost=cost)
