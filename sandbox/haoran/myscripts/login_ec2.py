import os
import argparse
from rllab import config
if config.BUCKET == "hrtang0":
    pem_file = "rllab-us-west-1"
elif config.BUCKET == "rllab-hrtang":
    pem_file = "hrtang-us-west-1"
parser = argparse.ArgumentParser()
parser.add_argument('ip',type=str,default='',nargs='?')
args = parser.parse_args()
command = "ssh ubuntu@{ip} -i private/key_pairs/{pem_file}.pem -o \"IdentitiesOnly yes\"".format(ip=args.ip, pem_file=pem_file)
print(command)
os.system(command)
