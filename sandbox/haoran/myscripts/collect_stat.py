import argparse
import os
import csv
import numpy as np
import json
import joblib
from rllab.misc.instrument import query_yes_no
from rllab.misc.console import colorize

parser = argparse.ArgumentParser()
parser.add_argument('data_paths', type=str, nargs='*')
parser.add_argument('--stat', type=str, default="RawReturnAverage", nargs='?')
parser.add_argument('--itr', type=int, default=-1, nargs='?')
parser.add_argument('--game', type=str, default="frostbite", nargs='?')
parser.add_argument('--output', type=str, default="output.pkl")
args = parser.parse_args()

csv_files = []
variant_files = []
for data_path in args.data_paths:
    if not data_path.endswith('/'):
        data_path += '/'
    print(data_path)
    dirname = os.path.dirname(data_path)
    subdirprefix = os.path.basename(data_path)
    for subdirname in os.listdir(dirname):
        path = os.path.join(dirname,subdirname)
        if os.path.isdir(path) and (subdirprefix in subdirname):
            csv_file = os.path.join(path,"progress.csv")
            variant_file = os.path.join(path,"variant.json")
            if os.path.exists(csv_file) and os.path.exists(variant_file):
                csv_files.append(csv_file)
                variant_files.append(variant_file)

stats = []
variants = []
for csv_file, variant_file in zip(csv_files, variant_files):
    with open(variant_file) as vf:
        variant = json.load(vf)
    if variant["game"] == args.game:
        # read data
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                continue
            data = dict()
            for key in fieldnames:
                data[key] = []
            for row in reader:
                for key in fieldnames:
                    value = row[key]
                    data[key].append(float(value))

        # compute total samples
        if args.stat in data:
            stats.append(data[args.stat][args.itr])
            variants.append(variant)
            print(csv_file)
stat = str(args.stat)
output = {
    stat: stats,
    "variants": variants
}
joblib.dump(output, args.output, compress=3)
