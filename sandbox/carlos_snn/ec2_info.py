instance_info = {
    "c4.large": dict(price=0.105, vCPU=2),
    "c4.xlarge": dict(price=0.209, vCPU=4),
    "c4.2xlarge": dict(price=0.419, vCPU=8),
    "c4.4xlarge": dict(price=0.838, vCPU=16),
    "m4.10xlarge": dict(price=2.394, vCPU=40),
    "m4.16xlarge": dict(price=1.5, vCPU=64),
    "c4.8xlarge": dict(price=1.675, vCPU=36),
    "g2.2xlarge": dict(price=0.65, vCPU=8),
}

subnet_info = {
    "us-west-1a": dict(SubnetID="subnet-76fc132e", Groups=["sg-343d0850"]),
    "us-west-1b": dict(SubnetID="subnet-2ba8734f", Groups=["sg-343d0850"]),
}
