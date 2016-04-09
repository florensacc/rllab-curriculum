from __future__ import print_function
from __future__ import absolute_import


# find all tnpg instances
import glob
import json
import boto3
import os
import sys


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")



ec2 = boto3.client(
    "ec2",
    region_name="us-east-1",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY", None),
    aws_secret_access_key=os.environ.get("AWS_ACCESS_SECRET", None),
)




folders = glob.glob("data/s3/icml-plot-rerun/*")

instance_names = []


for folder in folders:
    params = json.load(open(folder + "/params.json"))
    if params['json_args']['algo']['_name'] == 'rllab.algos.tnpg.TNPG':
        print(folder)
        instance_names.append(folder.split("/")[-1])



print(len(instance_names))

instances = ec2.describe_instances()
instances = [x['Instances'][0] for x in instances['Reservations']]
instances = [x for x in instances if x['Tags'][0]['Value'] in instance_names]
instances = [x for x in instances if x['State']['Name'] == 'running']

instance_ids = [x['InstanceId'] for x in instances]

if query_yes_no("Terminating %d instances..." % len(instance_ids)):
    response = ec2.terminate_instances(InstanceIds=instance_ids)
    print(response)

