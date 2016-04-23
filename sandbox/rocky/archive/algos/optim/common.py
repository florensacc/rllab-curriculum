import numpy as np

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

# Note: this method does NOT ensure that the resulting quadratic matrices are
# PSD! This is the responsibility for the individual optimization-based algorithms
def convexify(x, u, f_forward, f_cost, f_final_cost, grad_hints=None):
    
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

def sample_actions(lb, ub, n_actions):
    Du = len(lb)
    if np.any(np.isinf(lb)) or np.any(np.isinf(ub)):
        raise ValueError('Cannot sample unbounded actions')
    return np.random.rand(n_actions, Du) * (ub - lb)[None, :] + lb[None, :]

def reg_psd(Q):
    Q = 0.5*(Q + Q.T)
    if np.allclose(Q, np.diag(np.diag(Q))):
        # For diagonal matrix, regularization is easy
        return np.maximum(Q, 0)
    else:
        # Otherwise, do an eigendecomposition, and make all eigenvalues nonnegative
        w, v = np.linalg.eigh(Q)
        return v.dot(np.diag(np.maximum(w, 0))).dot(v.T)
