# this file allow you to delete all files (and hence folders) starting with a specific prefix

import argparse
import sys
import boto3

parser = argparse.ArgumentParser()
parser.add_argument('prefix',type=str, default='xxxxxxxxxxxxxxxxxxxxx',nargs='?')
args = parser.parse_args()

prefix = args.prefix
bucket = 'hrtang0'

client = boto3.client('s3')
keys = []
finished = False
for i in range(1000):
    objects = client.list_objects_v2(Bucket=bucket,Prefix=prefix)
    if 'Contents' not in list(objects.keys()):
        print("No files starting with prefix %s."%(prefix))
        print("Terminate.")
        sys.exit(1)
    for obj in objects['Contents']:
        if obj['Key'] in keys:
            finished = True
            break
        else:
            keys.append(obj['Key'])
    if finished:
        break
    

print("Ready to delete keys:")
for key in keys:
    print(key)
answer = input("Are you sure to delete {n_file} files? (y/n)".format(n_file=len(keys)))
while answer not in ['y','Y','n','N']:
    print("Please input y(Y) or n(N)")
    answer = input("Are you sure? (y/n)")
if answer in ['y','Y']: 
    client.delete_objects( 
        Bucket=bucket,
        Delete={
            'Objects':[
                {'Key':key} for key in keys
            ]
        }
    )
    print("Deletion complete.")
else:
    print("Abort deletion.")

