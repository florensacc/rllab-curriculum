# Source: Control-Limited Differential Dynamic Programming by Y Tassa et al.

import numpy as np
import scipy as sp
from numpy.linalg import LinAlgError
from misc.ext import extract

#class NonPositiveDefiniteError(Exception):
#    pass

def forward_pass(x0, uref, f_forward, f_cost, f_final_cost, K=None, xref=None):
    Dx = len(x0)
    N, Du = uref.shape
    x = np.zeros((N+1, Dx))
    uout = np.zeros((N, Du))
    cost = 0
    x[0] = x0
    for t in range(N):
        if K is not None and xref is not None:
            u = uref[t] + K[t].dot(x[t] - xref[t])
        else:
            u = uref[t]
        x[t+1] = f_forward(x[t], u)
        cost += f_cost(x[t], u)
        uout[t] = u
    cost += f_final_cost(x[N])
    return dict(x=x, cost=cost, u=uout)

def jacobian(x, f, eps=1e-2):
    Nx = len(x)
    Nf = len(f(x))
    J = np.zeros((Nf, Nx))
    eyex = np.eye(Nx)
    for dx in range(Nx):
        xp = x + eyex[dx] * eps
        xn = x - eyex[dx] * eps
        J[:, dx] = (f(xp) - f(xn)) / (2*eps)
    return J

def grad(x, f, eps=1e-2):
    Nx = len(x)
    g = np.zeros(Nx)
    eyex = np.eye(Nx)
    for dx in range(Nx):
        xp = x + eyex[dx] * eps
        xn = x - eyex[dx] * eps
        g[dx] = (f(xp) - f(xn)) / (2*eps)#scaled_eps)
    return g

def linearize(x, u, f_forward, f_cost, f_final_cost, grad_hints=None):
    
    Dx = x.shape[1]
    N, Du = u.shape
    fx = np.zeros((N, Dx, Dx))
    fu = np.zeros((N, Dx, Du))
    cx = np.zeros((N+1, Dx))
    cu = np.zeros((N, Du))
    cxx = np.zeros((N+1, Dx, Dx))
    cuu = np.zeros((N, Du, Du))
    cxu = np.zeros((N, Dx, Du))

    if grad_hints is None:
        grad_hints = {}

    df_dx = grad_hints.get('df_dx', lambda x0, u0: jacobian(x0, lambda x: f_forward(x, u0)))
    df_du = grad_hints.get('df_du', lambda x0, u0: jacobian(u0, lambda u: f_forward(x0, u)))
    dc_dx = grad_hints.get('dc_dx', lambda x0, u0: grad(x0, lambda x: f_cost(x, u0)))
    dc_du = grad_hints.get('dc_du', lambda x0, u0: grad(u0, lambda u: f_cost(x0, u)))
    dc_dxx = grad_hints.get('dc_dxx', lambda x0, u0: jacobian(x0, lambda x: grad(x, lambda x: f_cost(x, u0))))
    dc_dxu = grad_hints.get('dc_dxu', lambda x0, u0: jacobian(u0, lambda u: grad(x0, lambda x: f_cost(x, u))))
    dc_duu = grad_hints.get('dc_duu', lambda x0, u0: jacobian(u0, lambda u: grad(u0, lambda u: f_cost(x0, u))))
    dcf_dx = grad_hints.get('dcf_dx', lambda x0: grad(x0, f_final_cost))
    dcf_dxx = grad_hints.get('dcf_dxx', lambda x0: jacobian(x0, lambda x: grad(x, f_final_cost)))

    for k in range(N):
        fx[k] = df_dx(x[k], u[k])
        fu[k] = df_du(x[k], u[k])
        cx[k] = dc_dx(x[k], u[k])
        cu[k] = dc_du(x[k], u[k])
        cxx[k] = dc_dxx(x[k], u[k])
        cxu[k] = dc_dxu(x[k], u[k])
        cuu[k] = dc_duu(x[k], u[k])
        
    cx[N] = dcf_dx(x[N])
    cxx[N] = dcf_dxx(x[N])
    return dict(fx=fx, fu=fu, cx=cx, cu=cu, cxx=cxx, cxu=cxu, cuu=cuu)

def backward_pass(x, u, f_forward, f_cost, f_final_cost, grad_hints, reg):
    Dx = x.shape[1]
    N, Du = u.shape
    print 'linearizing...'
    fx, fu, cx, cu, cxx, cxu, cuu = extract(
        linearize(x, u, f_forward, f_cost, f_final_cost),
        "fx", "fu", "cx", "cu", "cxx", "cxu", "cuu"
    )
    print 'linearized'

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
        Qux = cxu[t].T + fu[t].T.dot(Vxx[t+1]).dot(fx[t])
        Quu[t] = cuu[t] + fu[t].T.dot(Vxx[t+1]).dot(fu[t]) + reg*np.eye(cuu[t].shape[0])
        try:
            U_Quu = np.linalg.cholesky(Quu[t])
            L_Quu = U_Quu.T
            k[t] = -sp.linalg.solve_triangular(U_Quu, sp.linalg.solve_triangular(L_Quu, Qu, lower=True))
            K[t] = -sp.linalg.solve_triangular(U_Quu, sp.linalg.solve_triangular(L_Quu, Qux, lower=True))
        except LinAlgError as e:#Exception as e:
            print 'error in %d' % t
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
                print 'backward pass...'
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
                print 'expected decrease is less than min reduction: decreasing alpha to %f' % alpha
        if fwd_succeeded:
            lambda_ = min(lambda_, 1) / lambda_scale_factor#, lambda_min)
            if lambda_ < lambda_min:
                lambda_ = lambda_min#0
            u = unew
            if abs(cost - cnew) / abs(cost) < rel_tol or abs(cost - cnew) < abs_tol:
                print 'no cost improvement'
                break
            cost = cnew
        else:
            lambda_ = max(lambda_, 1) * lambda_scale_factor
            if lambda_ > lambda_max:
                print("Cannot improve objective even with large regularization")
                break
        yield dict(u=u, K=K, k=k, x=x, Quu=Quu)
        print 'lambda:', lambda_
    yield dict(u=u, K=K, k=k, x=x, Quu=Quu)
