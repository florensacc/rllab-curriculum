import sys
import argparse
from rllab import config
import os
import boto3

parser = argparse.ArgumentParser()
parser.add_argument('--exp_prefix',type=str,default='trpo-momentum',nargs='?')
parser.add_argument('--log_file',type=str,default='myscripts/experiments/exp_001.log',nargs='?')
args = parser.parse_args()

file_list = ['progress.csv','params.json']
bucket = 'hrtang'

# grab a list of files to download


remote_root = config.AWS_S3_PATH
local_root = os.path.join(config.LOG_DIR,"s3")

with open(args.log_file,'r') as f:
    lines = f.readlines()
    for line in lines:
        exp_name = line.split("\"")[1]
        remote_dir = os.path.join(remote_root, args.exp_prefix, exp_name)
        local_dir = os.path.join(local_root, args.exp_prefix, exp_name)

        for file_name in file_list:
            remote_file = os.path.join(remote_dir,file_name)
            local_file = os.path.join(local_dir,file_name)
            command = """
                aws s3 cp {remote_file} {local_file}  
                """.format(
                    remote_file=remote_file,
                    local_file=local_file,
                )
            os.system(command)
        sys.exit(0)
            
