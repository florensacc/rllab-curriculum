from __future__ import division
import numpy as np
from collections import deque
import scipy.optimize as opt

class InverseHessianPairs(object):
    """
    LBFGS (inverse) Hessian approximation based on rotating list of pairs (step, delta gradient)
    that are assumed to approximately satisfy secant equation.
    """
    def __init__(self,max_num_pairs):
        self.syrhos = deque([],max_num_pairs) #pylint: disable=E1121
    def add(self,s,y):
        rho = 1./y.dot(s)
        if rho < 0: 
            print "WARNING: rho < 0" 
        self.syrhos.append((s,y,rho))
    def mvp(self,g):
        """
        Matrix-vector product        
        Nocedal & Wright Algorithm 7.4
        uses H0 = alpha*I, where alpha = <s,y>/<y,y>
        """
        assert len(self.syrhos) > 0

        q = g.copy()
        alphas = np.empty(len(self.syrhos))
        for (i,(s,y,rho)) in reversed(list(enumerate(self.syrhos))):
            alphas[i] = alpha = rho*s.dot(q)
            q -= alpha*y

        s,y,rho = self.syrhos[-1]
        ydoty = y.dot(y)
        sdoty = s.dot(y)
        gamma = sdoty/ydoty

        r = gamma*q

        for (i,(s,y,rho)) in enumerate(self.syrhos):
            beta = rho * y.dot(r)
            r += s * (alphas[i] - beta)

        return r

    
def lbfgs(f,fgrad,x0,maxiter=100,max_corr=25,grad_norm_tol=1e-9, ihp=None,ls_criteria="strong_wolfe"):
    """
    LBFGS algorithm as described by Nocedal & Wright
    In fact it gives numerically identical answers to L-BFGS-B on some test problems.
    """
    x = x0.copy()
    yield x
    if ihp is None: ihp = InverseHessianPairs(max_corr)
    oldg = fgrad(x)
    if ls_criteria=="armijo": fval = f(x)
    p = -oldg/np.linalg.norm(oldg)

    iter_count = 0
    while True:
        g=None
        if ls_criteria == "strong_wolfe":
            alpha_star, _, _, fval, _, g = opt.line_search(f,fgrad,x,p,oldg)
        elif ls_criteria == "armijo":
            import scipy.optimize.linesearch
            alpha_star = None
            alpha0 = 1.0
            alpha_star,_,fval=scipy.optimize.linesearch.line_search_armijo(f,x,p,oldg,fval,alpha0=alpha0)
        else:
            raise NotImplementedError

        if alpha_star is None:
            print("lbfgs line search failed after %i iterations!"%iter_count)
            # asdf
            break
        #assert np.isfinite(fval)
        s = alpha_star * p
        x += s
        yield x

        iter_count += 1
        
        if iter_count  >= maxiter:
            break

        if g is None: 
            # print("line search didn't give us a gradient. calculating")
            g = fgrad(x)

        if np.linalg.norm(g) < grad_norm_tol:
            break


        y = g - oldg
        ihp.add( s,y )
        p = ihp.mvp(-g)
        oldg = g



def test_lbfgs():
    print "*** Testing LBFGS ***"

    np.random.seed(0)
    Q = np.random.randn(100,10)
    H = Q.T.dot(Q)
    b = np.random.randn(10)
    f = lambda x: .5*x.dot(H.dot(x)) + b.dot(x)
    fgrad = lambda x: H.dot(x) + b

    x0 = np.random.randn(10)


    maxiter=5
    soln = opt.minimize(f, x0, method='L-BFGS-B',jac=fgrad,options=dict(maxiter=maxiter))
    x_scipy = soln['x']
    def last(seq):
        elem=None
        for elem in seq: pass
        return elem
    x_my = last(lbfgs(f, fgrad, x0, maxiter=maxiter+2))

    assert np.allclose(x_my, x_scipy)


if __name__ == "__main__":
    test_lbfgs()



