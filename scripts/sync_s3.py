import sys
sys.path.append('.')
from rllab import config
import os
import argparse
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, default=None, nargs='?')
    parser.add_argument('--dry', action='store_true', default=False)
    parser.add_argument('--bare', action='store_true', default=False)
    parser.add_argument('--noitr', action='store_true', default=False)
    parser.add_argument('--nohtml', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', default=False)
    args = parser.parse_args()
    remote_dir = config.AWS_S3_PATH
    local_dir = os.path.join(config.LOG_DIR, "s3")
    if args.folder:
        remote_dir = os.path.join(remote_dir, args.folder)
        local_dir = os.path.join(local_dir, args.folder)
    command = """aws s3 sync {remote_dir} {local_dir} --content-type "UTF-8" """.format(local_dir=local_dir, remote_dir=remote_dir)
    if args.noitr:
        command += """ --exclude '*itr*' """
    if args.nohtml:
        command += """ --exclude '*.html' """
    if args.bare:
        command += """ --exclude '*' --include '*.csv' --include '*.json' --include '*.html' """
    elif not args.all:
        command += """ --exclude '*stdout.log' --exclude '*stdouterr.log'  """
    if args.dry:
        print(command)
    else:
        os.system(command)
