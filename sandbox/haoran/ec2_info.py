instance_info = {
    "c4.large": dict(price=0.105, vCPU=2),
    "c4.xlarge": dict(price=0.209, vCPU=4),
    "c4.2xlarge": dict(price=0.419, vCPU=8),
    "c4.4xlarge": dict(price=0.838, vCPU=16),
    "m4.10xlarge": dict(price=2.394,vCPU=40),
    "c4.8xlarge": dict(price=1.675,vCPU=36),
    "g2.2xlarge": dict(price=0.65, vCPU=8),
}

subnet_info = {
    # in Trevor's account
    # "us-west-1c": dict(SubnetID="subnet-0fbfe456",Groups=["sg-83d85ee7"]),
    # "us-west-1a": dict(SubnetID="subnet-5927ee3d",Groups=["sg-b8f2b9dc"]),

    # OpenAI-Alpha
    # "us-west-1a": dict(SubnetID="subnet-03eb6e5a", Groups=["sg-960f08f2"]),
    # "us-west-1b": dict(SubnetID="subnet-f7296d92", Groups=["sg-960f08f2"]),

    # My own
    "us-west-1b": dict(SubnetID="subnet-fe0ee1a6", Groups=["sg-dd0700b9"]),
    "us-west-1c": dict(SubnetID="subnet-6ee43f0a", Groups=["sg-dd0700b9"]),
}
