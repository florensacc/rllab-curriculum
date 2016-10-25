import argparse
import os
import json
from rllab.misc.instrument import query_yes_no
from rllab.misc.console import colorize

parser = argparse.ArgumentParser()
parser.add_argument('prefix', type=str, default='??????',nargs='?')
parser.add_argument('--param', type=str, default='', nargs='?')
parser.add_argument('--old', type=str, default='', nargs='?')
parser.add_argument('--new', type=str, default='', nargs='?')
parser.add_argument('--yes', default=False, action='store_true')
args = parser.parse_args()

json_files = []
dirname = os.path.dirname(args.prefix)
subdirprefix = os.path.basename(args.prefix)
for subdirname in os.listdir(dirname):
    path = os.path.join(dirname,subdirname)
    if os.path.isdir(path) and (subdirprefix in subdirname):
        json_file = os.path.join(path,"variant.json")
        if os.path.exists(json_file):
            json_files.append(json_file)

for json_file in json_files:
    print("File: %s"%(json_file))
    confirm = False
    with open(json_file,"r") as f:
        params = json.load(f)
        if args.param in params.keys():
            old_value = params[args.param]
            value_type = type(old_value)
            if args.old == '' or value_type(args.old) != old_value:
                continue
            if args.yes:
                confirm = True
            else:
                new_value = value_type(args.new)
                confirm = query_yes_no("Replace {name}={old_value} by {new_value}?".format(
                    name=args.param,
                    old_value=old_value,
                    new_value=new_value,
                ))
            if confirm:
                params[args.param] = new_value 
    if confirm:
        with open(json_file,"w") as f:
            json.dump(params,f)
            print(colorize("Changes made and saved.","yellow"))


