#!/usr/bin/env python

import numpy as np

from rllab.misc import logger

from sandbox.sandy.adversarial.fgsm import fgsm_perturbation_linf, \
        fgsm_perturbation_l2, fgsm_perturbation_l1
from sandbox.sandy.adversarial.io_util import save_rollout_step
from sandbox.sandy.misc.util import get_softmax

BUFFER = 10

def get_action_probs_k_a3c(algo, obs, k, target_env, printout=True):
    # Get action probabilities for next *k* timesteps
    import chainer
    from chainer import functions as F

    # Set state of algo.cur_env to be identical to target env's state for rollouts,
    # especially *same ALE state*
    algo.cur_env.save_state()
    assert algo.cur_env.equiv_to(target_env)

    statevar = chainer.Variable(np.expand_dims(algo.cur_agent.preprocess(obs), 0))
    action_prob_k = np.zeros((k+1,algo.cur_agent.model.pi.n_actions))

    # Deterministically rolls out model to calculate J(\theta,x,y)
    prev_h, prev_c = algo.cur_agent.model.lstm.h, algo.cur_agent.model.lstm.c
    
    for i in range(k+1):  # +1 since we're including the current time step
        out = algo.cur_agent.model.lstm(algo.cur_agent.model.head(statevar))
        action_prob = algo.cur_agent.model.pi.compute_logits(out).data
        if np.abs(np.sum(action_prob)-1) > 1e-5:
            action_prob = get_softmax(action_prob)
        action_prob_k[i,:] = action_prob

        action = np.argmax(action_prob, axis=len(action_prob.shape)-1)
        if printout:
            print("Action:", action)
        algo.cur_env.receive_action(action)
        statevar = chainer.Variable(np.expand_dims(algo.cur_agent.preprocess(algo.cur_env.observation), 0))

    # Restore hidden state of lstm
    algo.cur_agent.model.lstm.h, algo.cur_agent.model.lstm.c = prev_h, prev_c
    algo.cur_env.restore_state()
    assert algo.cur_env.equiv_to(target_env)

    if printout:
        print(action_prob_k)
    return action_prob_k

def get_action_probs_k(algo, obs, adv_obs, k, target_env, printout=True):
    # Returns (action_prob_orig, action_prob_adv) where each is a k x action_dim
    # matrix
    algo_name = type(algo).__name__
    if algo_name in ['TRPO', 'ParallelTRPO']:
        raise NotImplementedError
    elif algo_name in ['A3CALE']:
        if printout:
            print("Actions without perturbation:")
        action_probs_orig = get_action_probs_k_a3c(algo, obs, k, target_env, printout=printout)
        if printout:
            print("Actions with perturbation:")
        action_probs_adv = get_action_probs_k_a3c(algo, adv_obs, k, target_env, printout=printout)
        return (action_probs_orig, action_probs_adv)
    elif algo_name in ['DQNAlgo']:
        raise NotImplementedError
    else:
        print("Algorithm type " + algo_name + " is not supported.")
        raise NotImplementedError

def get_grad_x_k_a3c(obs, algo, k, target_env, lambdas=None):
    # Obtains gradient for current and next k timesteps, for the loss function
    # that encourages keeping all argmax actions the same except for the
    # last time step's
    # lambdas - weights on the loss at each time step; length should be k+1

    import chainer
    from chainer import functions as F

    if lambdas is not None:
        assert len(lambdas) == k+1
    else:
        lambdas = [1]*k + [k]

    algo.cur_env.save_state()
    assert algo.cur_env.equiv_to(target_env)

    # Deterministically rolls out model to calculate J(\theta,x,y)
    prev_h, prev_c = algo.cur_agent.model.lstm.h, algo.cur_agent.model.lstm.c

    statevars = [None]*(k+1)
    statevars[0] = chainer.Variable(np.expand_dims(algo.cur_agent.preprocess(obs), 0))
    logits = [None]*(k+1)
    max_logits = [None]*(k+1)
    ce_loss = 0
    
    for i in range(k+1):  # +1 since we're including the current time step
        out = algo.cur_agent.model.lstm(algo.cur_agent.model.head(statevars[i]))
        logits[i] = algo.cur_agent.model.pi.compute_logits(out)
        action = np.argmax(logits[i].data, axis=1)
        algo.cur_env.receive_action(action)
        if i < k:
            statevars[i+1] = chainer.Variable(np.expand_dims( \
                    algo.cur_agent.preprocess(algo.cur_env.observation), 0))
        max_logits[i] = F.broadcast_to(F.max(logits[i]), (1,len(logits[i])))

        # Calculate loss between predicted action distribution and the action distribution
        # that places all weight on the argmax action
        ce_loss_step = F.log(1.0 / F.sum(F.exp(logits[i] - max_logits[i])))
        ce_loss_step *= lambdas[i]
        if i == k:
            ce_loss_step *= -1  # Want the argmax action to be *different* k steps in future
        ce_loss += ce_loss_step

    # The next three lines that clear gradients are probably not necessary (since
    # the Variables' gradients get initialized to None), but there just in case
    for x in [ce_loss] + statevars + logits +  max_logits:
        x.cleargrad()
    algo.cur_agent.model.cleargrads()

    ce_loss.backward(retain_grad=True)
    grad_x = np.array(statevars[0].grad[0])  # statevars[0].grad is 4D (first dim = # batches)

    # Restore hidden state of lstm
    algo.cur_agent.model.lstm.h, algo.cur_agent.model.lstm.c = prev_h, prev_c
    algo.cur_env.restore_state()
    assert algo.cur_env.equiv_to(target_env)

    # For debugging:
    #print("A3C:", abs(grad_x).max(), abs(grad_x).sum() / grad_x.size, ce_loss.data)
    return grad_x

def get_grad_x_k(obs, algo, k, target_env, lambdas=None):
    algo_name = type(algo).__name__
    if algo_name in ['TRPO', 'ParallelTRPO']:
        raise NotImplementedError
    elif algo_name in ['A3CALE']:
        return get_grad_x_k_a3c(obs, algo, k, target_env, lambdas=lambdas)
    elif algo_name in ['DQNAlgo']:
        raise NotImplementedError
    else:
        print("Algorithm type " + algo_name + " is not supported.")
        raise NotImplementedError

def get_stepsize(stepsizes, itr):
    if itr < len(stepsizes):
        return stepsizes[itr]
    return stepsizes[-1]

def compute_sleeper_cost(action_probs_orig, action_probs_adv):
    k = len(action_probs_orig)-1
    sleeper_cost = 0
    for i in range(k):
        if action_probs_orig[i] != action_probs_adv[i]:
            sleeper_cost += 1

    if action_probs_orig[k] == action_probs_adv[k]:  # Priority is making this different
        sleeper_cost += 100
    return sleeper_cost

def compute_adv_obs(obs, algo, k, target_env, norm, fgsm_eps, obs_min, obs_max, \
                    dual_descent_stepsizes=None, max_iter=10, init_lambda=None):
    # TODO: Possibly try other initial values for lambda
    # TODO: Scale lambdas to sum to 1?
    if init_lambda is None:
        init_lambda = np.array([1.0]*k + [float(k)])
    else:
        assert len(init_lambda) == k+1
    all_lambdas = [init_lambda]
    best_cost = float("inf")
    best_vals = None
    for _ in range(max_iter):
        # Calculate \grad_x J(\theta, x, y)
        #print("Trying lambdas:", all_lambdas[-1])
        grad_x = get_grad_x_k(obs, algo, k, target_env, lambdas=all_lambdas[-1])
        grad_x_current = grad_x[-1,:,:]

        if norm == 'l-inf':
            eta, unscaled_eta = fgsm_perturbation_linf(grad_x_current, fgsm_eps)
        elif norm == 'l2':
            eta, unscaled_eta = fgsm_perturbation_l2(grad_x_current, fgsm_eps)
        elif norm == 'l1':
            eta, unscaled_eta = fgsm_perturbation_l1(grad_x_current, fgsm_eps, \
                                                     obs[-1,:,:], obs_min, obs_max)
        else:
            raise NotImplementedError

        # The computed perturbation is stored in eta: x_adversarial = x + eta
        # Only current frame can be changed by adversary (i.e., the last frame)
        adv_obs = np.array(obs)
        adv_obs[-1,:,:] += eta
        # Clip pixels to be within range [obs_min, obs_max]
        adv_obs = np.minimum(obs_max, np.maximum(obs_min, adv_obs))

        # Calculate action probabilities before and after perturbation
        action_probs = get_action_probs_k(algo, obs, adv_obs, k, target_env, printout=False)

        if dual_descent_stepsizes is None:
            break

        # Update lambdas
        action_probs_orig, action_probs_adv = action_probs
        action_probs_orig = np.argmax(action_probs_orig, axis=1)
        action_probs_adv = np.argmax(action_probs_adv, axis=1)
        assert len(action_probs_orig) == k+1 and len(action_probs_adv) == k+1
        cost = compute_sleeper_cost(action_probs_orig, action_probs_adv)
        if cost == 0:
            print("success")
            best_cost = cost
            best_vals = (np.array(eta), np.array(unscaled_eta), np.array(adv_obs), \
                        (np.array(action_probs[0]),np.array(action_probs[1])), np.array(all_lambdas[-1]), cost)
            break
        elif cost < best_cost:
            best_cost = cost
            best_vals = (np.array(eta), np.array(unscaled_eta), np.array(adv_obs), \
                        (np.array(action_probs[0]),np.array(action_probs[1])), np.array(all_lambdas[-1]), cost)

        new_lambdas = np.array(all_lambdas[-1])
        alpha = get_stepsize(dual_descent_stepsizes, len(all_lambdas)-1)
        for i in range(k+1):
            if (i < k and action_probs_orig[i] != action_probs_adv[i]) or \
               (i == k and action_probs_orig[i] == action_probs_adv[i]):
                new_lambdas[i] += alpha
        all_lambdas.append(new_lambdas)

    if best_cost == 0:
        logger.record_tabular("Success:", True)
    else:
        logger.record_tabular("Success", False)
    logger.record_tabular("ActionsChosen", np.argmax(best_vals[3][0],axis=1))
    logger.record_tabular("ActionsChosenAdv", np.argmax(best_vals[3][1],axis=1))
    logger.record_tabular("Lambdas", best_vals[4])
    #return eta, unscaled_eta, adv_obs, action_probs, all_lambdas[-1]
    print("action probs:", np.argmax(best_vals[3][0],axis=1), np.argmax(best_vals[3][1],axis=1))
    print("lambdas:", best_vals[4])
    print("all lambdas:", all_lambdas)
    return best_vals, all_lambdas

def fgsm_sleeper_perturbation(obs, info, algo, **kwargs):
    # Apply fast gradient sign method (FGSM), only at time step t (specified
    # in kwargs). J(\theta, x, y) is cost on keeping the current and next k-1
    # (k specified in kwargs) actions the same, and making the one after that
    # different
    # 
    # For l-inf norm,
    #     \eta =  \epsilon * sign(\grad_x J(\theta, x, y))
    #         where \theta = policy params, x = obs, y = action
    # For l2 norm,
    #     \eta = (\epsilon / ||\grad_x J(\theta, x, y)||_2) * \grad_x J(\theta, x, y)
    # For l1 norm,
    #     going down list of indices i ranked by |\grad_x J(\theta, x, y)|_i,
    #     maximally perturb \eta_i (to obs_min or obs_max, depending on sign
    #     of (\grad_x J(\theta, x, y))_i; have 'budget' of \epsilon total perturbation

    import chainer

    try:
        t = kwargs['t']
        k = kwargs['k']
        fgsm_eps = kwargs['fgsm_eps']
        norm = kwargs['norm']
        obs_min = kwargs['obs_min']
        obs_max = kwargs['obs_max']
        output_h5 = kwargs.get('output_h5', None)
        results = kwargs.get('results', None)
        dual_descent_stepsizes = kwargs.get("dual_descent_stepsizes", None)
        init_lambda = kwargs.get("init_lambda", None)
    except KeyError:
        print("FGSM Sleeper requires the following inputs: t, k, fgsm_eps, norm, obs_min, obs_max")
        raise

    if hasattr(algo, 'env'):
        env = algo.env
    elif hasattr(algo, 'cur_env'):
        env = algo.cur_env
    else:
        raise NotImplementedError

    if obs is None and info is None:  # Hacky way to get access to LSTM states
        adversary_done = (len(env.actions_taken) > t + k + BUFFER)
        if hasattr(algo.cur_agent.model, 'lstm'):
            return algo.cur_agent.model.lstm.c.data, algo.cur_agent.model.lstm.h.data, adversary_done
        else:
            return None, None, adversary_done

    try:
        target_env = info['env']
        target_t = target_env.t
    except KeyError:
        print("FGSM Sleeper requires the following inputs in info: t, env")
        raise

    if target_t == 0:
        # On first call to fgsm_perturbation, make sure the adversary is
        # initialized correctly
        #print("Copying from target env to adv env")
        env.init_copy_from(target_env)
        assert len(target_env.actions_taken) == 0
    else:
        # Take the last action taken by target_env
        env.step(target_env.actions_taken[-1])

    assert env.equiv_to(target_env)

    # Only perturb input at time t
    if target_t != t:
        adv_obs = np.array(obs)
    else:
        print("TIME:", t)
        (eta, unscaled_eta, adv_obs, action_probs, lambdas, cost), all_lambdas = \
                compute_adv_obs(obs, algo, k, target_env, norm, fgsm_eps, \
                                obs_min, obs_max, dual_descent_stepsizes=dual_descent_stepsizes, \
                                init_lambda=init_lambda)
        if output_h5 is not None:
            to_save = dict(t=t, k=k, lambdas=lambdas, cost=cost, all_lambdas=all_lambdas)
            save_rollout_step(output_h5, eta, unscaled_eta, obs[-1,:,:], \
                              adv_obs[-1,:,:], action_probs, **to_save)

    # Update hidden state of algo, if it's an LSTM
    if type(algo).__name__ == "A3CALE":
        if hasattr(algo.cur_agent.model, 'lstm'):
            assert algo.cur_agent.model.skip_unchain
            statevar = chainer.Variable(np.expand_dims(algo.cur_agent.preprocess(adv_obs), 0))
            out = algo.cur_agent.model.lstm(algo.cur_agent.model.head(statevar))

    env.insert_adv_obs(adv_obs)
    return adv_obs
