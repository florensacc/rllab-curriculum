import boto3
import os

ec2 = boto3.client(
    "ec2",
    region_name="us-east-1",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY", None),
    aws_secret_access_key=os.environ.get("AWS_ACCESS_SECRET", None),
)

instances = ec2.describe_instances()

instances = [x['Instances'][0] for x in instances['Reservations']]
instances = [x for x in instances if x['Tags'][0]['Value'].startswith('dpg')]
instances = [x for x in instances if x['State']['Name'] == 'running']


for instance in instances:
    ip = instance['PublicIpAddress']
    name = instance['Tags'][0]['Value']
    # import ipdb; ipdb.set_trace()
    command = ("""
    scp -o StrictHostKeyChecking=no -r ubuntu@{ip}:/Users/dementrock/research/rllab/data/local/{name}/progress.csv data/s3/dpg-new-search/{name}/progress.csv
    """.format(ip=ip, name=name))
    print(command)
    os.system(command)

