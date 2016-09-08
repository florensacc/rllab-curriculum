# The implementation largely referenced TrajOpt
# Let's first use a collocation formulation 

import numpy as np
import scipy as sp
from misc.ext import extract
from gurobipy import *
from algo.optim.common import convexify, forward_pass, reg_psd
import operator

__all__ = [
        "solve"
]

def compute_cost(x, xref, u, merit, f_cost, f_final_cost):
    loss = 0
    N = len(x) - 1
    loss = f_final_cost(x[N])
    for t in range(N):
        loss += f_cost(x[t], u[t])
    return loss + merit * compute_violation(x, xref)

def compute_violation(x, xref):
    vio = 0
    N = len(x) - 1
    for t in range(N):
        vio += np.sum(np.abs(x[t+1] - xref[t+1]))
    return vio

# Gurobipy helpers
def ip(a, b):
    return LinExpr(a, b)

def quad_form(xs, coeff):
    Nx = len(xs)
    expr = QuadExpr()
    expr.addTerms(coeff.reshape(-1), [y for x in xs for y in [x] * Nx], [y for _ in range(Nx) for y in xs])
    return expr

def solve(
        x0,
        uinit,
        f_forward,
        f_cost,
        f_final_cost,
        state_bounds=None,
        action_bounds=None,
        grad_hints=None,
        improve_ratio_threshold=0.25,
        trust_shrink_ratio=0.6,
        trust_expand_ratio=1.5,
        max_merit_itr=5,
        merit_increase_ratio=10,
        min_trust_box_size=1e-4,
        initial_trust_box_size=0.1,
        min_model_improve=1e-4,
        ):

    N = uinit.shape[0]
    xinit = np.tile(x0.reshape(1, -1), (N+1, 1))
    Du = uinit.shape[1]
    Dx = len(x0)

    x = np.array(xinit)
    u = np.array(uinit)

    if state_bounds is None:
        state_bounds = -np.inf*np.ones(Dx), np.inf*np.ones(Dx)
    if action_bounds is None:
        action_bounds = -np.inf*np.ones(Dx), np.inf*np.ones(Dx)

    sco_itr = 0

    merit = 1

    for merit_itr in range(max_merit_itr):
        trust_box_size = initial_trust_box_size
        
        xref = [None] + [f_forward(x[t], u[t]) for t in range(N)]
        before_cost = compute_cost(x, xref, u, merit, f_cost, f_final_cost)

        dx = [[None] * Dx for _ in range(N+1)]
        du = [[None] * Du for _ in range(N)]
        model = Model("sqp")

        model.setParam(GRB.param.OutputFlag, 0)

        xlb, xub = state_bounds
        ulb, uub = action_bounds

        within_merit_itr = 0

        # xlb <= x + dx <= xub
        for t in range(N+1):
            for k in range(Dx):
                #print 'lb: %e, ub: %e' % (max(xlb[k], -GRB.INFINITY), min(xub[k], GRB.INFINITY))
                dx[t][k] = model.addVar(lb=xlb[k]-x[t][k], ub=xub[k]-x[t][k], name='dx_%d_%d' % (t, k))
        for t in range(N):
            for k in range(Du):
                du[t][k] = model.addVar(lb=ulb[k]-u[t][k], ub=uub[k]-u[t][k], name='du_%d_%d' % (t, k))
        norm_aux = [[None] * Dx for _ in range(N)]
        for t in range(N):
            for k in range(Dx):
                norm_aux[t][k] = model.addVar(lb=0, ub=GRB.INFINITY, name="norm_aux_%d_%d" % (t, k))
        model.update()

        for k in range(Dx):
            model.addConstr(dx[0][k] == 0)

        while True:
            # W/ this line: shooting; w/o: collocation
            # x = forward_pass(x[0], u, f_forward, f_cost, f_final_cost)["x"]

            xref = [None] + [f_forward(x[t], u[t]) for t in range(N)]

            within_merit_itr += 1
            if within_merit_itr > 10:
                print('within merit itr exceeded')
                break

            print('convexifying')
            fx, fu, cx, cu, cxx, cxu, cuu = extract(
                convexify(x, u, f_forward, f_cost, f_final_cost, grad_hints),
                "fx", "fu", "cx", "cu", "cxx", "cxu", "cuu"
            )
            print('convexified')

            loss = ip(cx[N], dx[N]) + quad_form(dx[N], 0.5*reg_psd(cxx[N])) + f_final_cost(x[N])

            aux_constrs = []
                
            for t in range(N):
                cquad = np.vstack([np.hstack([cxx[t], cxu[t]]), np.hstack([cxu[t].T, cuu[t]])])
                loss += ip(cx[t], dx[t]) + ip(cu[t], du[t]) + quad_form(dx[t] + du[t], 0.5*reg_psd(cquad)) + f_cost(x[t], u[t])
                for k in range(Dx):
                    loss += merit * norm_aux[t][k]
                    rhs = x[t+1,k] + dx[t+1][k] - xref[t+1][k] - ip(fx[t,k], dx[t]) - ip(fu[t,k], du[t])
                    aux_constrs.append(model.addConstr(norm_aux[t][k] >= rhs))
                    aux_constrs.append(model.addConstr(norm_aux[t][k] >= -rhs))

            model.setObjective(loss, GRB.MINIMIZE)
            
            trust_constraints = []
            
            no_improve = False

            before_cost = compute_cost(x, xref, u, merit, f_cost, f_final_cost)
            after_cost = None
            true_after_cost = None

            try:
                while trust_box_size > min_trust_box_size:

                    trust_constraints = []
                    
                    for t in range(1, N+1):
                        for k in range(Dx):
                            trust_constraints.append(model.addConstr(dx[t][k] <= trust_box_size))
                            trust_constraints.append(model.addConstr(dx[t][k] >= -trust_box_size))

                    for t in range(N):
                        for k in range(Du):
                            trust_constraints.append(model.addConstr(du[t][k] <= trust_box_size))
                            trust_constraints.append(model.addConstr(du[t][k] >= -trust_box_size))

                    model.optimize()
                    sco_itr += 1

                    for constr in trust_constraints:
                        model.remove(constr)

                    after_cost = model.objVal
                    
                    unew = np.zeros_like(u)
                    xnew = np.zeros_like(x)
                    for t, dut in enumerate(du):
                        for k, dutk in enumerate(dut):
                            unew[t][k] = u[t][k] + dutk.x
                    for t, dxt in enumerate(dx):
                        for k, dxtk in enumerate(dxt): 
                            xnew[t][k] = x[t][k] + dxtk.x
                    model_improve = before_cost - after_cost
                    if model_improve < -1e-5:
                        print("approximate merit function got worse (%f). (convexification is probably wrong to zeroth order)" % model_improve)
                    if model_improve < min_model_improve:
                        print("converged because improvement was small (%f < %f)" % (model_improve, min_model_improve))
                        no_improve = True
                        break
                    xnewref = [None] + [f_forward(xnew[t], unew[t]) for t in range(N)]


                    #xnew_shooting, _, _ = forward_pass(x[0], unew, f_forward, f_cost, f_final_cost)
                    true_after_cost = compute_cost(xnew, xnewref, unew, merit, f_cost, f_final_cost)
                    print("cost before: ", before_cost, "cost after: ", after_cost, "true cost after: ", true_after_cost)

                    #x = xnew
                    #u = unew
                    #xref = xnewref
                    #sco_itr = 101
                    #break


                    true_improve = before_cost - true_after_cost
                    improve_ratio = true_improve / model_improve
                    if improve_ratio >= improve_ratio_threshold:
                        trust_box_size *= trust_expand_ratio
                        print("trust box expanded to %f" % trust_box_size)
                        x = xnew
                        u = unew
                        xref = xnewref
                        break
                    else:
                        trust_box_size *= trust_shrink_ratio
                        print("trust box shrunk to %f" % trust_box_size)
                    if sco_itr > 100:
                        print("sco iteration exceeded")
                        break
                if sco_itr > 100:
                    print("sco iteration exceeded")
                    break
                if trust_box_size < min_trust_box_size:
                    print("converged because trust region is tiny")
                    break
                if no_improve:
                    break
            finally:
                for aux_constr in aux_constrs:
                    try:
                        model.remove(aux_constr)
                    except Exception as e:
                        pass
                aux_constrs = []
                yield dict(x=x, u=u, before_cost=before_cost, after_cost=after_cost, true_after_cost=true_after_cost)
        vio = compute_violation(x, xref)
        if vio < 1e-5:
            print('all constraints satisfied!')
            break
        elif merit_itr == max_merit_itr - 1:
            print('violation: %f' % vio)
        if sco_itr > 100:
            print("sco iteration exceeded")
            break

        merit *= merit_increase_ratio
