import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ip',type=str,default='',nargs='?')
args = parser.parse_args()
command = "ssh ubuntu@{ip} -i private/key_pairs/hrtang-us-west-1.pem -o \"IdentitiesOnly yes\"".format(ip=args.ip)
os.system(command)
