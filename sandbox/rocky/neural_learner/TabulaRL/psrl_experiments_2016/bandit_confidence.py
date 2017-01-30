'''
Script to output the width of Q-values confidence sets from Q-values

author: iosband@stanford.edu
'''

import numpy as np
import pandas as pd
import argparse
import sys

import environment
import finite_tabular_agents

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment
from shutil import copyfile

alg_dict = {'PSRL': finite_tabular_agents.PSRL,
            'PSRLunif': finite_tabular_agents.PSRLunif,
            'OptimisticPSRL': finite_tabular_agents.OptimisticPSRL,
            'GaussianPSRL': finite_tabular_agents.GaussianPSRL,
            'UCBVI': finite_tabular_agents.UCBVI,
            'BEB': finite_tabular_agents.BEB,
            'BOLT': finite_tabular_agents.BOLT,
            'UCRL2': finite_tabular_agents.UCRL2,
            'UCFH': finite_tabular_agents.UCFH,
            'EpsilonGreedy': finite_tabular_agents.EpsilonGreedy}




def outputConfidenceKnownR(alg, nextStateMul, nObs):
    '''
    Ouput the confidence set for a given algorithm, when P unknown.

    Args:
        alg - string for which algorithm to use
        nextStateMul - how many multiples of good/bad states there are
        nObs - how many observations split between the

    Returns:
        qMax - the 0.05 upper quantile
    '''
    nState = 1 + 2 * nextStateMul
    nextObs = nObs / float(nState - 1)

    # Make the environment
    env = environment.make_confidenceMDP(nextStateMul)

    # Make the feature extractor
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # Make the agent
    agent_constructor = alg_dict[alg]
    agent = agent_constructor(env.nState, env.nAction, env.epLen)

    # Letting the agent know the prior
    agent.R_prior[0, 0] = (0, 1e9)
    for s in range(1, nState):
        # Updating the P
        agent.P_prior[s, 0][s] += 1e9
        # Updating the rewards
        agent.R_prior[s, 0] = (s % 2, 1e9)

    for ep in range(nObs):
        # Reset the environment
        env.reset()
        agent.update_policy(ep)
        pContinue = 1
        while pContinue > 0:
            # Step through the episode
            h, oldState = f_ext.get_feat(env)
            action = 0
            reward, newState, pContinue = env.advance(action)
            agent.update_obs(oldState, action, reward, newState, pContinue, h)

    agent.update_policy()
    return agent.qVals[0, 0][0]


def outputConfidenceKnownP(alg, nextStateMul, nObs):
    '''
    Ouput the confidence set for a given algorithm, when R unknown.

    Args:
        alg - string for which algorithm to use
        nextStateMul - how many multiples of good/bad states there are
        nObs - how many observations split between the

    Returns:
        qMax - the 0.05 upper quantile
    '''
    nState = 1 + 2 * nextStateMul
    nextObs = nObs / float(nState - 1)

    # Make the environment
    env = environment.make_confidenceMDP(nextStateMul)

    # Make the feature extractor
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # Make the agent
    agent_constructor = alg_dict[alg]
    agent = agent_constructor(env.nState, env.nAction, env.epLen)

    # Letting the agent know the transitions, but not the rewards
    agent.R_prior[0, 0] = (0, 1e9)
    agent.P_prior[0, 0][0] = 0
    for s in range(1, env.nState):
        agent.P_prior[0, 0][s] += 1e9
        agent.P_prior[s, 0][s] += 1e9

    for ep in range(nObs):
        # Reset the environment
        env.reset()
        agent.update_policy(ep)
        pContinue = 1
        while pContinue > 0:
            # Step through the episode
            h, oldState = f_ext.get_feat(env)
            action = 0
            reward, newState, pContinue = env.advance(action)
            agent.update_obs(oldState, action, reward, newState, pContinue, h)

    agent.update_policy()
    return agent.qVals[0, 0][0]


def outputConfidenceH(alg, epLen, nObs):
    '''
    Ouput the confidence set for a given algorithm, when epLen changes.

    Args:
        alg - string for which algorithm to use
        nextStateMul - how many multiples of good/bad states there are
        nObs - how many observations split between the

    Returns:
        qMax - the 0.05 upper quantile
    '''
    # Make the environment
    env = environment.make_HconfidenceMDP(epLen)

    # Make the feature extractor
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # Make the agent
    agent_constructor = alg_dict[alg]
    agent = agent_constructor(env.nState, env.nAction, env.epLen)


    for ep in range(nObs):
        # Reset the environment
        env.reset()
        agent.update_policy(ep)
        pContinue = 1
        while pContinue > 0:
            # Step through the episode
            h, oldState = f_ext.get_feat(env)
            action = 0
            reward, newState, pContinue = env.advance(action)
            agent.update_obs(oldState, action, reward, newState, pContinue, h)

    agent.update_policy()
    return agent.qVals[0, 0][0]


def run_bandit_confidence(seed, a, mul, targetPath='tmp.csv'):
    '''
    Runs all three bandit experiments
    '''
    results = []
    np.random.seed(seed)
    nObs = 1000

    qMax = outputConfidenceKnownR(a, mul, nObs)
    results.append({'alg': a, 'mul': mul, 'epLen': -1, 'seed': seed,
                    'qMax': qMax, 'nObs': nObs, 'experiment': 'known_r'})

    qMax = outputConfidenceKnownP(a, mul, nObs)
    results.append({'alg': a, 'mul': mul, 'epLen': -1, 'seed': seed,
                    'qMax': qMax, 'nObs': nObs, 'experiment': 'known_p'})

    qMax = outputConfidenceH(a, mul, nObs)
    results.append({'alg': a, 'mul': -1, 'epLen': mul, 'seed': seed,
                    'qMax': qMax, 'nObs': nObs, 'experiment': 'epLen'})

    dt = pd.DataFrame(results)
    dt.to_csv('tmp.csv', index=False, float_format='%.2f')
    copyfile('tmp.csv', targetPath)
    print('********************************************')
    print('SUCCESS')
    print('********************************************')





