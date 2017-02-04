from rllab import config
import os
import json
import boto3

from rllab.misc.instrument import query_yes_no
from sandbox.rocky.cirrascale.bin.ec2ctl import get_all_instances, get_clients

exp_prefix = "fetch-analogy-5-5"

variant_filter = dict(
    # n_boxes={2,3,4},
    # batch_size={128,256,512}
    # hidden_sizes=[[512,512]],#512,512,512], [1024,1024]]
    n_train_tasks=[8],
    # noise_levels=[[0.], [0.,0.001]]
)

dir = os.path.join(config.PROJECT_PATH, "data/s3/{}".format(exp_prefix))

subfolders = os.listdir(dir)

exp_names = []

for folder in subfolders:
    dir_folder = os.path.join(dir, folder)
    variant_file = os.path.join(dir_folder, "variant.json")
    with open(variant_file, 'r') as f:
        variant_content = json.load(f)
        passed = True
        for key, values in variant_filter.items():
            if variant_content.get(key, None) not in values:#[key]
                passed = False
                break
        if passed:
            exp_names.append(variant_content["exp_name"])

to_kill_ids = {}
to_kill = []

for instance in get_all_instances():
    if 'Tags' in instance:
        tags = {t['Key']: t['Value'] for t in instance['Tags']}
        if tags['Name'] in exp_names:
            instance_id = instance['InstanceId']
            region = instance['Region']
            if region not in to_kill_ids:
                to_kill_ids[region] = []
            to_kill_ids[region].append(instance_id)
            to_kill.append(tags['Name'])

print("This will kill the following jobs:")
print(", ".join(sorted(to_kill)))
if query_yes_no(question="Proceed?", default="no"):
    for client in get_clients():
        print("Terminating instances in region", client.region)
        ids = to_kill_ids.get(client.region, [])
        if len(ids) > 0:
            client.terminate_instances(
                InstanceIds=to_kill_ids.get(client.region, [])
            )
