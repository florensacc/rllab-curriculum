#!/usr/bin/env python

""" Prints out all experiment names from the given directory, split by a
    particular criteria (e.g., input image size).
"""

import argparse
import json
import os
import os.path as osp

def get_value_in_json(data, sort_key):
    sort_key = sort_key.split(":")
    val = data
    for k in sort_key:
        val = val[k]
    return val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_parent_dir', type=str)
    # List of keys for accessing the sort-by value in each json object
    parser.add_argument('--sort_by_key', type=str, default="json_args:env:img_height")
    args = parser.parse_args()

    exp_dirs = os.listdir(args.exp_parent_dir)
    sorted_exp = {}
    for d in exp_dirs:
        data_file = open(osp.join(os.getcwd(),args.exp_parent_dir,d,'params.json')).read()
        data = json.loads(data_file)
        val = get_value_in_json(data, args.sort_by_key)
        
        if val not in sorted_exp:
            sorted_exp[val] = []
        sorted_exp[val].append(d)

    for val in sorted_exp:
        print(str(val) + ": " + " ".join(sorted_exp[val]))

if __name__ == "__main__":
    main()
