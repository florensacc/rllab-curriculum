from rllab import config
import os
import json
import boto3

dir = os.path.join(config.PROJECT_PATH, "data/s3/doom-maze-26")

subfolders = os.listdir(dir)

regions = ['us-east-1', 'us-west-1', 'us-west-2']
clients = [
    boto3.client(
        "ec2",
        region_name=region,
        aws_access_key_id=config.AWS_ACCESS_KEY,
        aws_secret_access_key=config.AWS_ACCESS_SECRET,
    )
    for region in regions
]

# for folder in subfolders:
#     dir_folder = os.path.join(dir, folder)
#     variant_file = os.path.join(dir_folder, "variant.json")
#     with open(variant_file, 'r') as f:
#         variant_content = json.load(f)
        # if int(variant_content['difficulty']) != 1:
        # exp_name = variant_content["exp_name"]
exp_name = "mab-9_2016_10_19_21_33_55_0034"
for client in clients:
    results = client.describe_instances(
        Filters=[
            {
                'Name': 'tag-value',
                'Values': [
                    exp_name,
                ]
            },
        ],
    )
    if len(results['Reservations']) > 0:
        # import ipdb; ipdb.set_trace()
        instance = results['Reservations'][0]['Instances'][0]
        ip_addr = instance['PublicIpAddress']
        i
        command = "ssh -t rocky@{gpu_host} 'cd /local_home/rocky/rllab-workdir/{job_id} && exec bash -l'".format(
            gpu_host=job['gpu_host'], job_id=job_id)
        print(command)
        os.system(command)
        import ipdb; ipdb.set_trace()
        # instance_id = instance['InstanceId']
        # print("Terminating instance {0} for exp {1}".format(instance_id, exp_name))
        # client.terminate_instances(
        #     InstanceIds=[
        #         instance_id,
        #     ]
        # )
