import argparse
import os
import csv
import numpy as np
from rllab.misc.instrument import query_yes_no
from rllab.misc.console import colorize

parser = argparse.ArgumentParser()
parser.add_argument('prefix', type=str, default='??????',nargs='?')
parser.add_argument('--param', type=str, default='', nargs='?')
parser.add_argument('--type', type=str, default='str', nargs='?')
parser.add_argument('--old', type=str, default='', nargs='?')
parser.add_argument('--new', type=str, default='', nargs='?')
parser.add_argument('--yes', default=False, action='store_true')
args = parser.parse_args()

csv_files = []
dirname = os.path.dirname(args.prefix)
subdirprefix = os.path.basename(args.prefix)
for subdirname in os.listdir(dirname):
    path = os.path.join(dirname,subdirname)
    if os.path.isdir(path) and (subdirprefix in subdirname):
        csv_file = os.path.join(path,"progress.csv")
        if os.path.exists(csv_file):
            csv_files.append(csv_file)

for csv_file in csv_files:
    # read data
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = dict()
        for key in fieldnames:
            data[key] = []
        for row in reader:
            for key in fieldnames:
                value = row[key]
                data[key].append(float(value))

    # compute total samples
    assert "NumSamples" in data.keys()
    data["TotalSamples"] = np.cumsum(data["NumSamples"])
    if "TotalSamples" not in fieldnames:
        fieldnames.append("TotalSamples")
                
    with open(csv_file,"w") as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        T = len(data["NumSamples"])
        for t in range(T):
            row = {key: data[key][t] for key in fieldnames}
            writer.writerow(row)    
    print("Modified %s"%(csv_file))
