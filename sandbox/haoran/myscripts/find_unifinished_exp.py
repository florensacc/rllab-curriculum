"""
Show a list of unfinished experiments.
output format:
folder_name, completed_iterations, total_iterations
"""

import json
import argparse
import sys,csv,os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=str, nargs='*')
    parser.add_argument("--prefix",type=str,nargs='?',default="???")
    args = parser.parse_args(sys.argv[1:])

    # load all folders following a prefix
    if args.prefix != "???":
        args.data_paths = []
        dirname = os.path.dirname(args.prefix)
        subdirprefix = os.path.basename(args.prefix)
        for subdirname in os.listdir(dirname):
            path = os.path.join(dirname,subdirname)
            if os.path.isdir(path) and (subdirprefix in subdirname):
                args.data_paths.append(path)

    unfinished_folders = []
    for path in args.data_paths:
        for folder, dirs, files  in os.walk(path):
            if "params.json" in files:
                with open(os.path.join(folder,"params.json")) as f:
                    params = json.load(f)
                    n_itr = params["json_args"]["algo"]["n_itr"]
                with open(os.path.join(folder,"progress.csv")) as f:
                    progress = csv.reader(f)
                    i = 0
                    for row in progress:
                        i += 1
                if (i-1) < n_itr:
                    unfinished_folders.append(folder)
                    print(folder, (i-1), n_itr)


