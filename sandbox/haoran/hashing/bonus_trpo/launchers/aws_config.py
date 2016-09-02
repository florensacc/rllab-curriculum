"""
Best practices
1. Distribute between machines
2. If an instance type is near capacity, don't submit any more requests.
"""

from rllab import config

def configure(instance_type):

if "g" in instance_type:
    use_gpu = True

# gpus need other dockers
if use_gpu:
    config.DOCKER_IMAGE = "tsukuyomi2044/rllab_gpu"
else:
    config.DOCKER_IMAGE = "dementrock/rllab-shared"


config.AWS_INSTANCE_TYPE = instance_type

# bid price = price_coeff * on-demand prices
# n_parallel = #vCPU / 2
# see: https://aws.amazon.com/ec2/pricing/?nc1=h_ls
if instance_type == "m4.large":
    price = 0.12
    n_parallel=1
elif instance_type == "m4.xlarge"
    price = 0.24
    n_parallel=2
elif "ec2_m4_2x" in mode:
    price = 0.48
    n_parallel=4
elif "ec2_c4" in mode:
    config.AWS_SPOT_PRICE = '0.105'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    n_parallel=1
elif "ec2_c4_x" in mode:
    config.AWS_SPOT_PRICE = '0.209'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    n_parallel=2
elif "ec2_c4_2x" in mode:
    config.AWS_SPOT_PRICE = '0.419'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    n_parallel=4
elif "ec2_g2" in mode:
    config.AWS_SPOT_PRICE = '1.5'
    config.DOCKER_IMAGE = "tsukuyomi2044/rllab_gpu"
