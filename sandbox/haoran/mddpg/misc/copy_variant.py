"""
If an experiment resumes another experiment, this script helps to
    update variant.json by copying params from the resumed exp.
This is only a temporary solution.
"""
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('prefix', type=str, default='??????',nargs='?', help="""
    e.g. "../exp-000/exp-000_swimmer" will operate on all swimmer folders,
    Alternatively, use ""../exp-000/". The final "/" is necessary.
""")
args = parser.parse_args()

paths = []
dirname = os.path.dirname(args.prefix) # ../exp-000
subdirprefix = os.path.basename(args.prefix) # exp-000_swimmer
for subdirname in os.listdir(dirname): # exp-000_swimmer_20170101
    path = os.path.join(dirname,subdirname) # ../exp-000/exp-000_swimmer_20170101
    if os.path.isdir(path) and (subdirname.startswith(subdirprefix)):
        # non-empty and starts with the prefix exp-000_swimmer
        paths.append(path)

for path in paths:
    variant_file = os.path.join(path, "variant.json")
    if os.path.exists(variant_file):
        os.system("cp %s %s.bak"%(variant_file, variant_file))
        with open(variant_file) as vf:
            variant = json.load(vf)
        ref_variant_file = os.path.join(
            'data/s3',
            variant["exp_info"]["exp_prefix"],
            variant["exp_info"]["exp_name"],
            "variant.json",
        )
        if os.path.exists(ref_variant_file):
            with open(ref_variant_file) as vf2:
                ref_variant = json.load(vf2)
            for k, v in ref_variant.items():
                if k not in variant:
                    variant["exp_info"][k] = v
            with open(variant_file, "w") as vf:
                json.dump(variant, vf)
            print("Copied variants from %s to %s"%(ref_variant_file, variant_file))
        else:
            print("%s does not exist!"%(ref_variant_file))
