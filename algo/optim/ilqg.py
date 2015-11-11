# Source: Control-Limited Differential Dynamic Programming by Y Tassa et al.

import numpy as np
import scipy as sp
from numpy.linalg import LinAlgError
from misc.ext import extract

#class NonPositiveDefiniteError(Exception):
#    pass

def forward_pass(x0, uref, sysdyn, cost_func, final_cost_func, K=None, xref=None):
    Dx = len(x0)
    N, Du = uref.shape
    x = np.zeros((N+1, Dx))
    uout = np.zeros((N, Du))
    cost = 0
    x[0] = x0
    for t in range(N):
        if K and xref:
            u = uref[t] + K[t].dot(x[t] - xref[t])
        else:
            u = uref[t]
        x[t+1] = sysdyn(x[t], u)
        cost += cost_func(x[t], u)
        uout[t] = u
    cost += final_cost_func(x[N])
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

def linearize(x, u, sysdyn, cost_func, final_cost_func):
    
    Dx = x.shape[1]
    N, Du = u.shape
    fx = np.zeros((N, Dx, Dx))
    fu = np.zeros((N, Dx, Du))
    cx = np.zeros((N+1, Dx))
    cu = np.zeros((N, Du))
    cxx = np.zeros((N+1, Dx, Dx))
    cuu = np.zeros((N, Du, Du))
    cxu = np.zeros((N, Dx, Du))

    for k in range(N):
        fx[k] = jacobian(x[k], lambda x: sysdyn(x, u[k]))
        fu[k] = jacobian(u[k], lambda u: sysdyn(x[k], u))
        cx[k] = grad(x[k], lambda x: cost_func(x, u[k]))
        cu[k] = grad(u[k], lambda u: cost_func(x[k], u))
        cxu[k] = jacobian(u[k], lambda u: grad(x[k], lambda x: cost_func(x, u)))
        cuu[k] = jacobian(u[k], lambda u: grad(u, lambda u: cost_func(x[k], u)))
        
    cx[N] = grad(x[N], final_cost_func)
    cxx[N] = jacobian(x[N], lambda x: grad(x, final_cost_func))
    return dict(fx=fx, fu=fu, cx=cx, cu=cu, cxx=cxx, cxu=cxu, cuu=cuu)

def backward_pass(x, u, sysdyn, cost_func, final_cost_func, reg):
    Dx = x.shape[1]
    N, Du = u.shape
    print 'linearizing...'
    fx, fu, cx, cu, cxx, cxu, cuu = extract(
        linearize(x, u, sysdyn, cost_func, final_cost_func),
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

def solve(x0, uinit, sysdyn, cost_func, final_cost_func,
        max_iter=100,
        min_reduction=0,
        max_line_search_iter=8,
        lambda_init=1.0,
        lambda_scale_factor=10.0,
        lambda_max=1e6,
        lambda_min=1e-3,
        ):
    
    Dx = len(x0)
    Du = uinit.shape[1]
    N = len(uinit)
    
    u = uinit

    lambda_ = lambda_init
    
    for itr in range(max_iter):
        x, cost = extract(
            forward_pass(x0, u, sysdyn, cost_func, final_cost_func),
            "x", "cost"
        )
        print 'cost:', cost

        bwd_succeeded = False
        while not bwd_succeeded:
            try:
                print 'backward pass...'
                Vx, Vxx, k, K, dVx, dVxx, Quu = extract(
                    backward_pass(x, u, sysdyn, cost_func, final_cost_func, reg=lambda_),
                    "Vx", "Vxx", "k", "K", "dVx", "dVxx", "Quu"
                )
                bwd_succeeded = True
            except LinAlgError as e:#NonPositiveDefiniteError:
                print e
                lambda_ = max(lambda_ * lambda_scale_factor, lambda_min)
                print 'increasing lambda to %f' % lambda_
                if lambda_ > lambda_max:
                    break

        if not bwd_succeeded:
            print("Cannot find positive definition solution even with very large regularization")
            break

        #import ipdb; ipdb.set_trace()

        alpha = 1
        fwd_succeeded = False
        for _ in range(max_line_search_iter):
            print 'alpha: ', alpha
            # unew is different from u+alpha*k because of the feedback control term
            xnew, cnew, unew = extract(
                forward_pass(x0, u+alpha*k, sysdyn, cost_func, final_cost_func, K=K, xref=x),
                "x", "cost", "u"
            )
            print 'cnew:', cnew#u+alpha*k
            dcost = cost - cnew
            expected = -alpha*(dVx+alpha*dVxx)
            if expected < 0:
                raise ValueError('should not happen')
            if dcost / expected > min_reduction:
                fwd_succeeded = True
                break
            else:
                alpha = alpha * 0.5
        if fwd_succeeded:
            lambda_ = min(lambda_, 1) / lambda_scale_factor#, lambda_min)
            if lambda_ < lambda_min:
                lambda_ = 0
            u = unew
            if abs(cost - cnew) / abs(cost) < 1e-6:
                print 'no cost improvement'
                break
            cost = cnew
        else:
            lambda_ = max(lambda_, 1) * lambda_scale_factor
            if lambda_ > lambda_max:
                print("Cannot improve objective even with large regularization")
                break
        print 'lambda:', lambda_
    return dict(u=u, K=K, k=k, x=x, Quu=Quu)
