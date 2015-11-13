from mdp.locomotion_mdp import LocomotionMDP
import numpy as np
import scipy as sp
from gurobipy import *

def run_test():

    mdp = LocomotionMDP()
    x0, _ = mdp.reset()

    from algo.optim.ilqg import jacobian, grad, linearize, forward_pass

    def compute_cost(x, xref, u, merit, cost_func, final_cost_func):
        loss = 0
        N = x.shape[0] - 1
        loss = final_cost_func(x[N])
        for t in range(N):
            loss += cost_func(x[t], u[t])
            loss += merit * np.sum(np.abs(x[t+1] - xref[t+1]))
        #import ipdb; ipdb.set_trace()
        return loss

    def compute_violation(x, xref):
        vio = 0
        for t in range(N):
            vio += np.sum(np.abs(x[t+1] - xref[t+1]))
        return vio


    N = mdp.horizon
    x0, _ = mdp.reset()
    xinit = np.tile(x0.reshape(1, -1), (N+1, 1))#mdp.xinit
    Du = mdp.action_dim
    Dx = len(x0)
    #print Du
    uinit = (np.random.rand(N, Du)-0.5)*0.1#0
    #uinit[:, 0] = 1
    #uinit[:, 1] = 1
    #uinit = (np.random.rand(N, Du)-0.5)*2#0.1#0

    #u = uinit
    #mdp.demo(uinit)#[])#u)


    sysdyn = mdp.forward_dynamics
    cost_func = mdp.cost
    final_cost_func = mdp.final_cost
    x = np.array(xinit)
    u = np.array(uinit)
    improve_ratio_threshold = 0.25#0.25
    trust_shrink_ratio = 0.6
    trust_expand_ratio = 1.5
    max_merit_itr = 1#5
    merit_increase_ratio = 10
    min_trust_box_size = 1e-4
    min_model_improve = 1e-4

    # adaptive scaling config
    min_scaling = 5#1e-2
    max_scaling = 1e6
    decay_rate = 0.9

    x_scale = np.ones_like(x) * min_scaling
    u_scale = np.ones_like(u) * min_scaling
    scale_t = 0

    import operator
    def ip(a, b):
        return LinExpr(a, b)

    def quad_form(xs, coeff):
        Nx = len(xs)
        expr = QuadExpr()
        expr.addTerms(coeff.reshape(-1), [y for x in xs for y in [x] * Nx], [y for _ in range(Nx) for y in xs])
        return expr

    sco_itr = 0

    merit = 1#00#00#0.0
    # try shooting first
    for merit_itr in range(max_merit_itr):
        trust_box_size = 0.1
        xref = [None] + [sysdyn(x[t], u[t]) for t in range(N)]
        before_cost = compute_cost(x, xref, u, merit, cost_func, final_cost_func)

        dx = [[None] * Dx for _ in range(N+1)]
        du = [[None] * Du for _ in range(N)]
        model = Model("sqp")

        model.setParam(GRB.param.OutputFlag, 0)

        xlb, xub = mdp.state_bounds

        # xlb <= x + dx <= xub
        for t in range(N+1):
            for k in range(Dx):
                #print 'lb: %e, ub: %e' % (max(xlb[k], -GRB.INFINITY), min(xub[k], GRB.INFINITY))
                dx[t][k] = model.addVar(lb=xlb[k]-x[t][k], ub=xub[k]-x[t][k], name='dx_%d_%d' % (t, k))
        for t in range(N):
            for k in range(Du):
                du[t][k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='du_%d_%d' % (t, k))
        norm_aux = [[None] * Dx for _ in range(N)]
        for t in range(N):
            for k in range(Dx):
                norm_aux[t][k] = model.addVar(lb=0, ub=GRB.INFINITY, name="norm_aux_%d_%d" % (t, k))
        model.update()

        for k in range(Dx):
            model.addConstr(dx[0][k] == 0)

        while True:
            # W/ this line: shooting; w/o: collocation
            # x, _, _ = forward_pass(x[0], u, sysdyn, cost_func, final_cost_func)

            mdp.demo(u, True)
            scale_t += 1
            x_scale = np.clip((decay_rate * x_scale + (1 - decay_rate) * x), min_scaling, max_scaling)
            u_scale = np.clip((decay_rate * u_scale + (1 - decay_rate) * u), min_scaling, max_scaling)

            xref = [None] + [sysdyn(x[t], u[t]) for t in range(N)]

            print 'linearizing'
            fx, fu, cx, cu, cxx, cxu, cuu = linearize(x, u, sysdyn, cost_func, final_cost_func)
            print 'linearized'
            
            loss = ip(cx[N], dx[N]) + quad_form(dx[N], cxx[N]) + final_cost_func(x[N])

            aux_constrs = []
                
            for t in range(N):
                cquad = np.vstack([np.hstack([cxx[t], cxu[t]]), np.hstack([cxu[t].T, cuu[t]])])
                loss += ip(cx[t], dx[t]) + ip(cu[t], du[t]) + quad_form(dx[t] + du[t], cquad) + cost_func(x[t], u[t])
                for k in range(Dx):
                    loss += merit * norm_aux[t][k]
                    rhs = x[t+1,k] + dx[t+1][k] - xref[t+1][k] - ip(fx[t,k], dx[t]) - ip(fu[t,k], du[t])
                    aux_constrs.append(model.addConstr(norm_aux[t][k] >= rhs))
                    aux_constrs.append(model.addConstr(norm_aux[t][k] >= -rhs))

            model.setObjective(loss, GRB.MINIMIZE)
            
            trust_constraints = []
            
            no_improve = False

            before_cost = compute_cost(x, xref, u, merit, cost_func, final_cost_func)

            try:
                while trust_box_size > min_trust_box_size:

                    trust_constraints = []
                    
                    for t in range(1, N+1):
                        for k in range(Dx):
                            if scale_t < 0:#>= 0:#< 0:#>= 0:#10:
                                size = trust_box_size * x_scale[t][k]
                            else:
                                size = trust_box_size
                            trust_constraints.append(model.addConstr(dx[t][k] <= size))
                            trust_constraints.append(model.addConstr(dx[t][k] >= -size))

                    for t in range(N):
                        for k in range(Du):
                            if scale_t < 0:#>= 0:#< 0:#>= 0:#10:
                                size = trust_box_size * u_scale[t][k]
                            else:
                                size = trust_box_size
                            trust_constraints.append(model.addConstr(du[t][k] <= size))
                            trust_constraints.append(model.addConstr(du[t][k] >= -size))

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
                        print "approximate merit function got worse (%f). (convexification is probably wrong to zeroth order)" % model_improve
                    if model_improve < min_model_improve:
                        print "converged because improvement was small (%f < %f)" % (model_improve, min_model_improve)
                        no_improve = True
                        break
                    xnewref = [None] + [sysdyn(xnew[t], unew[t]) for t in range(N)]


                    #xnew_shooting, _, _ = forward_pass(x[0], unew, sysdyn, cost_func, final_cost_func)
                    true_after_cost = compute_cost(xnew, xnewref, unew, merit, cost_func, final_cost_func)
                    print "cost before: ", before_cost, "cost after: ", after_cost, "true cost after: ", true_after_cost

                    #x = xnew
                    #u = unew
                    #xref = xnewref
                    #sco_itr = 101
                    #break


                    true_improve = before_cost - true_after_cost
                    improve_ratio = true_improve / model_improve
                    if improve_ratio >= improve_ratio_threshold:
                        trust_box_size *= trust_expand_ratio
                        print "trust box expanded to %f" % trust_box_size
                        x = xnew
                        u = unew
                        xref = xnewref
                        break
                    else:
                        trust_box_size *= trust_shrink_ratio
                        print "trust box shrunk to %f" % trust_box_size
                    if sco_itr > 100:
                        print "sco iteration exceeded"
                        break
                if sco_itr > 100:
                    print "sco iteration exceeded"
                    break
                if trust_box_size < min_trust_box_size:
                    print "converged because trust region is tiny"
                    break
                if no_improve:
                    break
            finally:
                for aux_constr in aux_constrs:
                    try:
                        model.remove(aux_constr)
                    except Exception as e:
                        pass
                        #print e
                aux_constrs = []
        vio = compute_violation(x, xref)
        if vio < 1e-5:
            print 'all constraints satisfied!'
            break
        elif merit_itr == max_merit_itr - 1:
            print 'violation: %f' % vio
        if sco_itr > 100:
            print "sco iteration exceeded"
            break

        merit *= merit_increase_ratio

    import ipdb; ipdb.set_trace()
    mdp.demo(u)

run_test()
#import hotshot, hotshot.stats
#prof = hotshot.Profile("sqp.prof")
#prof.runcall(run_test)#test.pystone.pystones)
#prof.close()
#stats = hotshot.stats.load("sqp.prof")
#stats.strip_dirs()
#stats.sort_stats('time', 'calls')
#stats.print_stats(20)
