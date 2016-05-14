from __future__ import print_function
from __future__ import absolute_import

import boto3
from rllab import config
from rllab.viskit.core import load_params, load_progress, flatten_dict
import os
import glob
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


if __name__ == "__main__":

    folders = glob.glob(os.path.join(config.PROJECT_PATH, "data/s3/async-ddpg-267final/*"))

    to_terminate = []

    for folder in folders:

        params_file_path = os.path.join(folder, "params.json")
        progress_file_path = os.path.join(folder, "progress.csv")
        log_file_path = os.path.join(folder, "stdout.log")

        params = load_params(params_file_path)
        progress = load_progress(progress_file_path)
        flat_params = flatten_dict(params)


        max_samples = flat_params["json_args.algo.max_samples"]
        actual_samples = int(progress['NSamples'][-1])

        machine_name = folder.split("/")[-1]

        if actual_samples >= max_samples:
            to_terminate.append(machine_name)
        else:
            try:
                log = open(log_file_path).read()
                if "NaN detected" in log:
                    to_terminate.append(machine_name)
            except IOError:
                continue

    print(len(to_terminate))

    ec2s = [
        boto3.client(
            "ec2",
            region_name="us-east-1",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY", None),
            aws_secret_access_key=os.environ.get("AWS_ACCESS_SECRET", None),
        ),
        boto3.client(
            "ec2",
            region_name="us-west-1",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY", None),
            aws_secret_access_key=os.environ.get("AWS_ACCESS_SECRET", None),
        ),
    ]

    for ec2 in ec2s:
        instances = ec2.describe_instances()
        instances = [x['Instances'][0] for x in instances['Reservations']]
        instances = [x for x in instances if x['Tags'][0]['Value'] in to_terminate]
        instances = [x for x in instances if x['State']['Name'] == 'running']

        instance_ids = [x['InstanceId'] for x in instances]

        if len(instance_ids) > 0:
            if query_yes_no("Terminating %d instances..." % len(instance_ids)):
                response = ec2.terminate_instances(InstanceIds=instance_ids)
                print(response)
