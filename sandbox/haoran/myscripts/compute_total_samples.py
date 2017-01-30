import argparse
import os
import csv
import numpy as np
from rllab.misc.instrument import query_yes_no
from rllab.misc.console import colorize

parser = argparse.ArgumentParser()
parser.add_argument('prefix', type=str, default='??????',nargs='?')
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
    assert "NumSamples" in data.keys()
    num_samples = data["NumSamples"]
    num_samples[num_samples=='nan'] = 0
    data["TotalSamples"] = np.cumsum(num_samples)
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
