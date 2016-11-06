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
from .bandit_confidence import run_bandit_confidence



if __name__ == '__main__':
    '''
    Run a tabular experiment according to command line arguments
    '''

    # Take in command line flags
    parser = argparse.ArgumentParser(description='Run tabular RL experiment')
    parser.add_argument('epLen', help='length of episode', type=int)
    parser.add_argument('gap', help='gap between best arm', type=float)
    parser.add_argument('alg', help='Agent constructor', type=str)
    parser.add_argument('scaling', help='scaling', type=float)
    parser.add_argument('seed', help='random seed', type=int)
    parser.add_argument('nEps', help='number of episodes', type=int)
    args = parser.parse_args()

    # Make a filename to identify flags
    fileName = ('bandit'
                + '_len=' + '%02.f' % args.epLen
                + '_gap=' + '%04.3f' % args.gap
                + '_alg=' + str(args.alg)
                + '_scal=' + '%03.2f' % args.scaling
                + '_seed=' + str(args.seed)
                + '.csv')

    folderName = './'
    targetPath = folderName + fileName
    print('******************************************************************')
    print(fileName)
    print('******************************************************************')

    # Run the experiment
    run_bandit_confidence(args.seed, args.alg, args.epLen, targetPath)

