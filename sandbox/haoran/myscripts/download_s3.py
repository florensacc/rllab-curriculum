# this file allow you to download all files (and hence folders) starting with a specific prefix

import argparse
import sys
import boto3
import re
import os
from rllab import config

parser = argparse.ArgumentParser()
parser.add_argument('bucket',type=str,default='hrtang',nargs='?')
parser.add_argument('--prefix',type=str, default='experiments',nargs='?')
parser.add_argument('--keyword',type=str,default='xxxxx',nargs='?')
args = parser.parse_args()

client = boto3.client('s3')
all_keys = []
matched_keys = []
finished = False
paginator = client.get_paginator('list_objects')
objects_iterator = paginator.paginate(Bucket=args.bucket,Prefix=args.prefix)

#if 'Contents' not in objects.keys():
#    print "No files starting with prefix %s."%(args.prefix)
#    print "Terminate."
#    sys.exit(1)

# find all matching files
for objects in objects_iterator:
    for obj in objects['Contents']:
        if obj['Key'] in all_keys: # stop sweeping
            finished = True
            break
        else:
            all_keys.append(obj['Key'])
            if args.keyword in obj['Key']:
                matched_keys.append(obj['Key'])

# last check of files
print("Ready to download keys:")
for key in matched_keys:
    print(key)
answer = input("Are you sure to download {n_file} files? (y/n)".format(n_file=len(matched_keys)))
while answer not in ['y','Y','n','N']:
    print("Please input y(Y) or n(N)")
    answer = input("Are you sure? (y/n)")

# start operation
if answer in ['y','Y']: 
    for key in matched_keys:
        key2 = key.split('experiments/')[1]
        local_file = os.path.join(config.LOG_DIR,"s3",key2)
        client.download_file(args.bucket,key,local_file)
    print("Download complete.")
else:
    print("Abort deletion.")

