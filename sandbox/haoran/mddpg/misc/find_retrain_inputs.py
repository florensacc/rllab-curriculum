import argparse
import os
import json

# exp_prefix="tuomas/vddpg/"
# snapshot_file = "itr_399.pkl"
# paths = []
# dirname = "data/s3/%s"%(exp_prefix)
# for subdirname in os.listdir(dirname):
#     path = os.path.join(dirname,subdirname)
#     if os.path.isdir(path):
#         paths.append(path)
#
# inputs = []
# for path in paths:
#     variant_file = os.path.join(path, "variant.json")
#     if os.path.exists(variant_file):
#         with open(variant_file) as vf:
#             v = json.load(vf)
#         snapshot_file_full = os.path.join(path, snapshot_file)
#         if v["alpha"] == 1 and os.path.exists(snapshot_file_full):
#             input = """
#                 dict(
#                     exp_prefix=\"{exp_prefix}\",
#                     exp_name=\"{exp_name}\",
#                     snapshot_file=\"{snapshot_file}\",
#                     env_name=\"{env_name}\",
#                     seed={seed}
#                 ),
#             """.format(
#                 exp_prefix=exp_prefix,
#                 exp_name=v["exp_name"],
#                 snapshot_file=snapshot_file,
#                 env_name=v["env_name"],
#                 seed=v["zzseed"],
#             )
#             inputs.append(input)
#
# output_file = "data/s3/%s/transfer_inputs.py"%(exp_prefix)
# with open(output_file,"w") as f:
#     for input in inputs:
#         f.write(input)
# print("Output to:", output_file)


exp_prefix="tuomas/vddpg/exp-000b"
snapshot_file = "itr_399.pkl"
paths = []
dirname = "data/s3/%s"%(exp_prefix)
for subdirname in os.listdir(dirname):
    path = os.path.join(dirname,subdirname)
    if os.path.isdir(path):
        paths.append(path)

inputs = []
for path in paths:
    variant_file = os.path.join(path, "variant.json")
    if os.path.exists(variant_file):
        with open(variant_file) as vf:
            v = json.load(vf)
        snapshot_file_full = os.path.join(path, snapshot_file)
        if v["alpha"] == 1 and os.path.exists(snapshot_file_full):
            input = """
                dict(
                    exp_prefix=\"{exp_prefix}\",
                    exp_name=\"{exp_name}\",
                    snapshot_file=\"{snapshot_file}\",
                    env_name=\"{env_name}\",
                    seed={seed}
                ),
            """.format(
                exp_prefix=exp_prefix,
                exp_name=v["exp_name"],
                snapshot_file=snapshot_file,
                env_name=v["env_name"],
                seed=v["zzseed"],
            )
            inputs.append(input)

output_file = "data/s3/%s/transfer_inputs.py"%(exp_prefix)
with open(output_file,"w") as f:
    for input in inputs:
        f.write(input)
print("Output to:", output_file)
