#!/usr/bin/env python

""" Regenerate videos for all saved rollouts in a particular directory
"""

import os
from sandbox.sandy.adversarial.vis import visualize_adversary

OUTPUT_DIR = "/home/shhuang/src/rllab-private/data/local/rollouts/exp001/exp001_20170113_200153_921814/"

def main():
    fnames = os.listdir(OUTPUT_DIR)
    for f in fnames:
        if '.h5' in f and 'fgsm' not in f:
            output_prefix = f.split('.')[0]
            visualize_adversary(os.path.join(OUTPUT_DIR,f), OUTPUT_DIR, output_prefix) 

if __name__ == "__main__":
    main()
