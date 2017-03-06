#!/usr/bin/env python

import numpy as np

from sandbox.sandy.adversarial.fgsm import fgsm_perturbation_linf, \
        fgsm_perturbation_l2, fgsm_perturbation_l1
from sandbox.sandy.adversarial.io_util import save_rollout_step
from sandbox.sandy.misc.util import get_softmax

def get_action_probs_k_a3c(algo, obs, k, target_env):
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
        print("Action:", action)
        algo.cur_env.receive_action(action)
        statevar = chainer.Variable(np.expand_dims(algo.cur_agent.preprocess(algo.cur_env.observation), 0))

    # Restore hidden state of lstm
    algo.cur_agent.model.lstm.h, algo.cur_agent.model.lstm.c = prev_h, prev_c
    algo.cur_env.restore_state()
    assert algo.cur_env.equiv_to(target_env)

    print(action_prob_k)
    return action_prob_k

def get_action_probs_k(algo, obs, adv_obs, k, target_env):
    # Returns (action_prob_orig, action_prob_adv) where each is a k x action_dim
    # matrix
    algo_name = type(algo).__name__
    if algo_name in ['TRPO', 'ParallelTRPO']:
        raise NotImplementedError
    elif algo_name in ['A3CALE']:
        print("Actions without perturbation:")
        action_probs_orig = get_action_probs_k_a3c(algo, obs, k, target_env)
        print("Actions with perturbation:")
        action_probs_adv = get_action_probs_k_a3c(algo, adv_obs, k, target_env)
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
    except KeyError:
        print("FGSM Sleeper requires the following inputs: t, k, fgsm_eps, norm, obs_min, obs_max")
        raise

    try:
        target_env = info['env']
        target_t = target_env.t
    except KeyError:
        print("FGSM Sleeper requires the following inputs in info: t, env")
        raise

    if hasattr(algo, 'env'):
        env = algo.env
    elif hasattr(algo, 'cur_env'):
        env = algo.cur_env
    else:
        raise NotImplementedError

    if target_t == 0:
        # On first call to fgsm_perturbation, make sure the adversary is
        # initialized correctly
        print("Copying from target env to adv env")
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
        # Calculate \grad_x J(\theta, x, y)
        grad_x = get_grad_x_k(obs, algo, k, target_env)
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
        action_probs = get_action_probs_k(algo, obs, adv_obs, k, target_env)

        if output_h5 is not None:
            to_save = dict(t=t, k=k)
            save_rollout_step(output_h5, eta, unscaled_eta, obs[-1,:,:], \
                              adv_obs[-1,:,:], action_probs, **to_save)

    # Update hidden state of algo, if it's an LSTM
    if type(algo).__name__ == "A3CALE":
        if hasattr(algo.cur_agent.model, 'lstm'):
            assert algo.cur_agent.model.skip_unchain
            statevar = chainer.Variable(np.expand_dims(algo.cur_agent.preprocess(adv_obs), 0))
            out = algo.cur_agent.model.lstm(algo.cur_agent.model.head(statevar))
    # TODO: Check to make sure hidden state of algo.cur_agent.model is same
    # as hidden state of target_algo.cur_agent.model

    env.insert_adv_obs(adv_obs)
    return adv_obs
