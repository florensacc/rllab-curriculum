# Source: Control-Limited Differential Dynamic Programming by Y Tassa et al.

import numpy as np
import scipy as sp
from numpy.linalg import LinAlgError

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
        if K is not None and xref is not None:
            u = uref[t] + K[t].dot(x[t] - xref[t])
        else:
            u = uref[t]
        x[t+1] = sysdyn(x[t], u)
        cost += cost_func(x[t], u)
        uout[t] = u
    cost += final_cost_func(x[N])
    return x, cost, uout

def jacobian(x, f, eps=1e-5):
    Nx = len(x)
    Nf = len(f(x))
    J = np.zeros((Nf, Nx))
    eyex = np.eye(Nx)
    for dx in range(Nx):
        #scaled_eps = max(eps, abs(x[dx]) * eps)
        xp = x + eyex[dx] * eps#scaled_eps#eps
        xn = x - eyex[dx] * eps#scaled_eps#eps
        J[:, dx] = (f(xp) - f(xn)) / (2*eps)#scaled_eps)#eps)
    return J

def grad(x, f, eps=1e-5):
    Nx = len(x)
    g = np.zeros(Nx)
    eyex = np.eye(Nx)
    for dx in range(Nx):
        #scaled_eps = max(eps, abs(x[dx]) * eps)
        #xp = x + eyex[dx] * scaled_eps#eps
        #xn = x - eyex[dx] * scaled_eps#eps

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
    return fx, fu, cx, cu, cxx, cxu, cuu

def backward_pass(x, u, sysdyn, cost_func, final_cost_func, reg):
    Dx = x.shape[1]
    N, Du = u.shape
    fx, fu, cx, cu, cxx, cxu, cuu = linearize(x, u, sysdyn, cost_func, final_cost_func)
    
    Vx = np.zeros((N+1, Dx))
    Vxx = np.zeros((N+1, Dx, Dx))
    Vx[N] = cx[N]
    Vxx[N] = cxx[N]
    k = np.zeros((N, Du))
    K = np.zeros((N, Du, Dx))
    
    # dVx and dVxx are used for computing the expected reduction in cost
    # dV ~ -alpha*Q_u.T*k - alpha^2*k.T*Q_uu*k
    dVx = 0
    dVxx = 0
    
    for t in range(N-1, -1, -1):
        Qx = cx[t] + fx[t].T.dot(Vx[t+1])
        Qu = cu[t] + fu[t].T.dot(Vx[t+1])
        Qxx = cxx[t] + fx[t].T.dot(Vxx[t+1]).dot(fx[t])
        Qxx = 0.5*(Qxx+Qxx.T)
        # TODO regularize
        Qux = cxu[t].T + fu[t].T.dot(Vxx[t+1]).dot(fx[t])

        Quu = cuu[t] + fu[t].T.dot(Vxx[t+1]).dot(fu[t]) + reg*np.eye(cuu[t].shape[0])
        #reg = 0
        #while True:
        #    try:
        #        Quu = 0.5*(Quu+Quu.T)
        #        U_Quu = np.linalg.cholesky(Quu)
        #        break
        #    except LinAlgError as e:
        #        import ipdb; ipdb.set_trace()
        #        if reg == 0:
        #            reg = 1
        #        else:
        #            reg = reg * 1.6
        #        print reg, t
        #        if reg > 1e6:
        #            import ipdb; ipdb.set_trace()
        #            raise

        # properly regularize Quu
        #u, s, v = np.linalg.svd(Quu)
        #s[s < 1e-6] = 1e-6
        #Quu = u.dot(np.diag(s)).dot(v.T)
        #np.linalg.eigvals(Quu))





        # TODO regularize
        try:
            U_Quu = np.linalg.cholesky(Quu)
            L_Quu = U_Quu.T
            k[t] = -sp.linalg.solve_triangular(U_Quu, sp.linalg.solve_triangular(L_Quu, Qu, lower=True))#np.linalg.inv(Quu)*Qu
            K[t] = -sp.linalg.solve_triangular(U_Quu, sp.linalg.solve_triangular(L_Quu, Qux, lower=True))#np.linalg.inv(Quu)*Qu
        #try:
        #    k[t] = -np.linalg.inv(Quu).dot(Qu)#sp.linalg.solve_triangular(U_Quu, sp.linalg.solve_triangular(L_Quu, Qu, lower=True))#np.linalg.inv(Quu)*Qu
        #    K[t] = -np.linalg.inv(Quu).dot(Qux)
        except LinAlgError as e:#Exception as e:
            print 'error in %d' % t
            #eigvals = np.linalg.eigvals(Quu)
            #import ipdb; ipdb.set_trace()
            #print eigvals
            #if np.all(eigvals > 0):
            #    import ipdb; ipdb.set_trace()
            #print 
            raise 
            #import ipdb; ipdb.set_trace()
        
        Vx[t] = Qx + Qux.T.dot(k[t])#- K[t].T.dot(Quu).dot(k[t])
        Vxx[t] = Qxx + Qux.T.dot(K[t])#- K[t].T.dot(Quu).dot(K[t])

        if np.any(abs(Vxx) > 1e10):
            raise LinAlgError('failed')
            #import ipdb; ipdb.set_trace()
        
        dVx += np.inner(k[t], Qu)
        dVxx += 0.5*k[t].T.dot(Quu).dot(k[t])
    return Vx, Vxx, k, K, dVx, dVxx

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
        x, cost, _ = forward_pass(x0, u, sysdyn, cost_func, final_cost_func)
        print cost

        bwd_succeeded = False
        while not bwd_succeeded:
            try:
                print 'backward pass...'
                Vx, Vxx, k, K, dVx, dVxx = backward_pass(x, u, sysdyn, cost_func, final_cost_func, reg=lambda_)
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

        alpha = 1
        fwd_succeeded = False
        for _ in range(max_line_search_iter):
            # unew is different from u+alpha*k because of the feedback control term
            xnew, cnew, unew = forward_pass(x0, u+alpha*k, sysdyn, cost_func, final_cost_func, K, x)
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
        print lambda_
    return u, K, x
