instance_info = {
    "c4.large": dict(price=0.105, vCPU=2),
    "c4.xlarge": dict(price=0.209, vCPU=4),
    "c4.2xlarge": dict(price=0.419, vCPU=8),
    "c4.4xlarge": dict(price=0.838, vCPU=16),
    "c4.8xlarge": dict(price=1.675, vCPU=36),
    "m4.large": dict(price=0.1, vCPU=2),
    "m4.xlarge": dict(price=0.1, vCPU=4),
    "m4.2xlarge": dict(price=0.1, vCPU=8),
    "m4.4xlarge": dict(price=0.1, vCPU=16),
    "m4.10xlarge": dict(price=2.394, vCPU=40),
    "m4.16xlarge": dict(price=1.5, vCPU=64),
    "g2.2xlarge": dict(price=0.65, vCPU=8),
}

subnet_info = {
    # N. California
    "us-west-1a": dict(SubnetID="subnet-76fc132e", Groups=["sg-343d0850"]),
    "us-west-1b": dict(SubnetID="subnet-2ba8734f", Groups=["sg-343d0850"]),
    # Oregon
    "us-west-2a": dict(SubnetID="subnet-a9675ccd", Groups=["sg-343d0850"]),
    "us-west-2b": dict(SubnetID="subnet-e5257793", Groups=["sg-343d0850"]),
    "us-west-2c": dict(SubnetID="subnet-9d3da9c5", Groups=["sg-343d0850"]),
    # N. Virginia
    "us-east-1a": dict(SubnetID="subnet-2f4de566", Groups=["sg-343d0850"]),
    "us-east-1b": dict(SubnetID="subnet-1c3dc247", Groups=["sg-343d0850"]),
    "us-east-1d": dict(SubnetID="subnet-d9a35bf4", Groups=["sg-343d0850"]),
    "us-east-1e": dict(SubnetID="subnet-bcc13680", Groups=["sg-343d0850"]),
    # Ohio
    "us-east-2a": dict(SubnetID="subnet-b6c73edf", Groups=["sg-343d0850"]),
    "us-east-2b": dict(SubnetID="subnet-2fc1d757", Groups=["sg-343d0850"]),
    "us-east-2c": dict(SubnetID="subnet-e8a196a2", Groups=["sg-343d0850"]),
}
