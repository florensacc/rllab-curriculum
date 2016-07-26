#!/usr/bin/env python
""" This script runs a pre-trained network with the game
visualization turned on.

Specify the network file first, then any other options you want
"""
import subprocess
import sys
import argparse


def run_watch(args):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-screen', dest="screen", default=True, action="store_false",
                        help="Don't show the screen. Only option that should come before the network")        
    parser.add_argument('networkfile', nargs=1,
                        help='Network file. Use "none" to test a newly created (ie random) network')
    parser.add_argument('--paper',default="nature",
                        help="Either nature or nips.") 
    parameters, unknown = parser.parse_known_args(args)

    exp_dir="models/temp2"    
    command = ['./run_nature.py', 
            '--steps-per-epoch', '0',
            '--experiment-directory',exp_dir,
            '--max-history','100000',
            '--epochs','1',
            '--steps-per-epoch','100000',
            '--test-length','1',
            '--epsilon-start','0.9'] 
    if parameters.networkfile[0].lower() != 'none':
        command.extend(['--nn-file', parameters.networkfile[0]])
    if parameters.screen:
        command.append('--display-screen')

    command += unknown
    p1 = subprocess.Popen(command)
    
    p1.wait()

    return 0

if __name__ == "__main__":
    sys.exit(run_watch(sys.argv[1:]))
