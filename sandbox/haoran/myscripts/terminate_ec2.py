# this file allow you to delete all files (and hence folders) starting with a specific prefix

import os
import json
import argparse
import sys
import boto3
from rllab import config

parser = argparse.ArgumentParser()
parser.add_argument('exp_prefix',type=str, default='xxxxxxxxxxxxxxxxxxxxx',nargs='?')
parser.add_argument('-y', default=False, action='store_true', help='force yes')
parser.add_argument('--v', type=str, default='', help='key=value')
args = parser.parse_args()

exp_prefix = args.exp_prefix
bucket = config.BUCKET

# grab all exp_name, instance_id corresponding to the give exp_prefix
# beward that sometimes you need a '/' at the end
client = boto3.client('ec2')
response = client.describe_instances(
    Filters=[
        {
            'Name': 'tag:exp_prefix',
            'Values': [
                exp_prefix,
            ],
        },
        {
            'Name': 'instance-state-name',
            'Values': [
                'running',
            ],
        }
    ]
)
instance_ids = []
exp_names = []
for reservation in response["Reservations"]:
    for instance in reservation["Instances"]:
        instance_ids.append(instance["InstanceId"])
        for tag in instance["Tags"]:
            if tag['Key'] == 'Name':
                exp_names.append(tag['Value'])

# find exps that match the variant
matched_exp_names = []
matched_instance_ids = []
for exp_name, instance_id in zip(exp_names, instance_ids):
    local_log_dir = os.path.join(
        config.LOG_DIR,
        "s3",
        exp_prefix.replace("_", "-"),
        exp_name,
    )
    variant_file = os.path.join(
        local_log_dir,
        "variant.json",
    )
    if not os.path.exists(variant_file):
        remote_log_dir = os.path.join(
            config.AWS_S3_PATH,
            exp_prefix.replace("_","-"),
            exp_name,
        )
        remote_variant_file = os.path.join(
            remote_log_dir,
            "variant.json",
        )
        os.system("""
            aws s3 cp {remote} {local}
        """.format(
            remote=variant_file,
            local=remote_variant_file,
        ))
    with open(variant_file) as vf:
        variant = json.load(vf)
        specified_variant = eval("dict(%s)"%(args.v))
        matches = [
            variant[key] == value
            for key, value in specified_variant.items()
        ]
        if matches.count(False) == 0:
            matched_exp_names.append(exp_name)
            matched_instance_ids.append(instance_id)
            print(exp_name, instance_id)

if args.y:
    answer = 'y'
else:
    answer = input("Are you sure to terminate {N} instances? (y/n)".format(
        N=len(matched_instance_ids),
    ))
    while answer not in ['y','Y','n','N']:
        print("Please input y(Y) or n(N)")
        answer = input("Are you sure? (y/n)")
if answer in ['y','Y']:
    response = client.terminate_instances(
        InstanceIds=matched_instance_ids,
    )
    print("Deletion complete.")
else:
    print("Abort deletion.")
