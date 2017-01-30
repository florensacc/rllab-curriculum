import argparse
import os
import csv
import numpy as np
import joblib
import sys
import gc
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from rllab.misc.instrument import query_yes_no
from rllab.misc.console import colorize
from sandbox.haoran.mddpg.algos.mddpg import MDDPG
from sandbox.haoran.myscripts.myutilities import get_true_env
from sandbox.haoran.mddpg.misc.sampler import MNNParallelSampler
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def compile_video(img_file_list, ani_file):
    # assume the image files are pre-sorted
    fig = plt.figure()
    ax = fig.add_subplot(111)
    example_img = plt.imread(img_file_list[0])
    I = ax.imshow(example_img)
    def update_img(img_file):
        img = plt.imread(img_file)
        I.set_data(img)
    fig.set_size_inches([10,10])
    fig.tight_layout()

    ani = animation.FuncAnimation(fig, update_img, img_file_list)
    writer = animation.writers['ffmpeg'](fps=5)
    ani.save(ani_file, writer=writer, dpi=200)
    return ani

def get_img_files(path, pattern, rank):
    import re
    file_list = []
    rank_list = []
    for filename in os.listdir(path):
        if re.match(pattern, filename):
            file_list.append(filename)
            rank_list.append(rank(filename))
    indices = np.argsort(rank_list)
    file_list = [os.path.join(path, file_list[ind]) for ind in indices]
    return file_list

parser = argparse.ArgumentParser()
parser.add_argument('prefix', type=str, default='??????',nargs='?', help="""
    e.g. "../exp-000/exp-000_swimmer" will operate on all swimmer folders,
    Alternatively, use ""../exp-000/". The final "/" is necessary.
""")
parser.add_argument('--path', type=str, default='', nargs='?', help="""
    Specify a single path to compile the video. Ignore prefix.
""")
parser.add_argument('--pattern', type=str, default='itr_.*_test_paths.png',
    nargs='?', help='Pattern for the image names')
parser.add_argument('--index', type=int, default=1, nargs='?',
    help='Position of the indexing part of the file name.')
parser.add_argument('--output', type=str, default='test_paths.mp4',
    nargs='?', help='Name of the output video')
args = parser.parse_args()

if args.path == '':
    paths = []
    dirname = os.path.dirname(args.prefix) # ../exp-000
    subdirprefix = os.path.basename(args.prefix) # exp-000_swimmer
    for subdirname in os.listdir(dirname): # exp-000_swimmer_20170101
        path = os.path.join(dirname,subdirname) # ../exp-000/exp-000_swimmer_20170101
        if os.path.isdir(path) and (subdirname.startswith(subdirprefix)):
            # non-empty and starts with the prefix exp-000_swimmer
            paths.append(path)
else:
    paths = [args.path]

for path in paths:
    rank = lambda filename: int(filename.split('_')[args.index])
    img_file_list = get_img_files(path, args.pattern, rank)
    ani_file = os.path.join(path, args.output)
    compile_video(img_file_list, ani_file)
    print("Generated %s"%(ani_file))
