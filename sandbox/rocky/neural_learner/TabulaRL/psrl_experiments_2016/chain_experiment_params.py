'''
Script to run tabular experiments in batch mode.

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



if __name__ == '__main__':
    '''
    Run a tabular experiment according to command line arguments
    '''

    # Take in command line flags
    parser = argparse.ArgumentParser(description='Run tabular RL experiment')
    parser.add_argument('chainLen', help='length of chain', type=int)
    parser.add_argument('alg', help='Agent constructor', type=str)
    parser.add_argument('alpha0', help='prior strength', type=float)
    parser.add_argument('tau', help='noise variance', type=float)
    parser.add_argument('scaling', help='scaling', type=float)
    parser.add_argument('seed', help='random seed', type=int)
    parser.add_argument('nEps', help='number of episodes', type=int)
    args = parser.parse_args()

    # Make a filename to identify flags
    fileName = ('chainExperiment'
                + '_len=' + '%03.f' % args.chainLen
                + '_alg=' + str(args.alg)
                + '_alpha0=' + '%04.2f' % args.alpha0
                + '_tau=' + '%04.4f' % args.tau
                + '_scal=' + '%03.2f' % args.scaling
                + '_seed=' + str(args.seed)
                + '.csv')

    folderName = './'
    targetPath = folderName + fileName
    print('******************************************************************')
    print(fileName)
    print('******************************************************************')

    # Make the environment
    env = environment.make_stochasticChain(args.chainLen)

    # Make the feature extractor
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # Make the agent
    alg_dict = {'PSRL': finite_tabular_agents.PSRL,
                'PSRLunif': finite_tabular_agents.PSRLunif,
                'OptimisticPSRL': finite_tabular_agents.OptimisticPSRL,
                'GaussianPSRL': finite_tabular_agents.GaussianPSRL,
                'UCBVI': finite_tabular_agents.UCBVI,
                'BEB': finite_tabular_agents.BEB,
                'BOLT': finite_tabular_agents.BOLT,
                'UCRL2': finite_tabular_agents.UCRL2,
                'UCFH': finite_tabular_agents.UCFH}
    agent_constructor = alg_dict[args.alg]

    agent = agent_constructor(env.nState, env.nAction, env.epLen,
                              alpha0=args.alpha0, tau=args.tau,
                              scaling=args.scaling)

    # Run the experiment
    run_finite_tabular_experiment(agent, env, f_ext, args.nEps, args.seed,
                        recFreq=100, fileFreq=1000, targetPath=targetPath)

