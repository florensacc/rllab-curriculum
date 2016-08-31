# delete all folders without 'progress.csv' or whose 'progress.csv' has too few lines

import argparse
import os
import sys
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('root',type=str, default='data/s3/trpo-cg',nargs='?')
parser.add_argument('--nlines',type=int,default=50,help='Threshold on number of lines.')
args = parser.parse_args()

def isdir(folder):
    return os.path.isdir(os.path.join(args.root,folder))

bad_folders = []
for folder in filter(isdir,os.listdir(args.root)):
    progress_file_name = os.path.join(args.root,folder,'progress.csv')
    if not os.path.isfile(progress_file_name):
        n_line = 0
    else:
        with open(progress_file_name,'r') as f:
            n_line = len(f.readlines())
    if n_line < args.nlines:
        bad_folders.append(folder)
        print folder + " n_line: %d"%(n_line)

answer = raw_input("Are you sure to delete {n_folder} folders? (y/n)".format(n_folder=len(bad_folders)))
while answer not in ['y','Y','n','N']:
    print "Please input y(Y) or n(N)"
    answer = raw_input("Are you sure? (y/n)")
if answer in ['y','Y']:
    for folder in bad_folders:
        shutil.rmtree(os.path.join(args.root,folder))
    print "Deleteion complete."
else:
    print "Abort deletion."
