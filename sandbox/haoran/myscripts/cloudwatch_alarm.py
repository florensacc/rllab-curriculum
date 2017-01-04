"""
This script tests whether EC2 Cloudwatch successfully sends a notification email if the instance status is wrong.
"""
from sandbox.haoran.ec2_info import instance_info, subnet_info
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab import config
import sys

stub(globals())
import boto3

mode = "ec2"
ec2_instance = "c4.large"
subnet = "us-west-1a"

if "local_docker" in mode:
    actual_mode = "local_docker"
elif "local" in mode:
    actual_mode = "local"
elif "ec2" in mode:
    actual_mode = "ec2"
    # configure instance
    info = instance_info[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"] /2)

    # choose subnet
    config.AWS_NETWORK_INTERFACES = [
        dict(
            SubnetId=subnet_info[subnet]["SubnetID"],
            Groups=subnet_info[subnet]["Groups"],
            DeviceIndex=0,
            AssociatePublicIpAddress=True,
        )
    ]

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)

ec2 = boto3.client(
    "ec2",
    region_name=config.AWS_REGION_NAME,
    aws_access_key_id=config.AWS_ACCESS_KEY,
    aws_secret_access_key=config.AWS_ACCESS_SECRET,
)
action = 'arn:aws:sns:us-west-1:994177516630:cpu_usage'
client = boto3.client('cloudwatch')
client.put_metric_alarm(
    AlarmName='low_cpu_usage',
    AlarmDescription='',
    ActionsEnabled=False,
    MetricName='CPUUtilization',
    Namespace='AWS/EC2',
    Statistic='Average',
    OKActions=[action],
    AlarmActions=[action],
    Dimensions=[],
    Period=60,
    EvaluationPeriods=1,
    Unit='Percent',
    Threshold=100,
    ComparisonOperator='LessThanOrEqualToThreshold',
)


run_experiment_lite(
    algo.train(),
    exp_prefix="cloudwatch",
    n_parallel=1,
    snapshot_mode="last",
    seed=0,
    mode=actual_mode,
    terminate_machine=False,
)
