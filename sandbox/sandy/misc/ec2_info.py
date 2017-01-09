from rllab import config
instance_info = {
    "c4.large": dict(price=0.02, vCPU=2),
    "c4.xlarge": dict(price=0.08, vCPU=4),
    "c4.2xlarge": dict(price=0.08, vCPU=8),
    "c4.4xlarge": dict(price=0.14, vCPU=16),
    "m4.10xlarge": dict(price=0.37,vCPU=40),
    "c4.8xlarge": dict(price=0.28,vCPU=36),
    "g2.2xlarge": dict(price=0.12, vCPU=8),
}

all_subnet_info = {
    'rllab-shhuang': {
        "us-west-1a": dict(
            SubnetID="subnet-f624ac92", Groups=["sg-01b87766"]), # sg-02b87765
            #SubnetID="subnet-17b73c73", Groups=["sg-01b87766"]), # sg-02b87765
        "us-west-1c": dict(
            SubnetID="subnet-7f904927", Groups=["sg-01b87766"]), # sg-02b87765
        "us-west-2a": dict(
            SubnetID="subnet-a8a48cdf", Groups=["sg-d9559ca1"]), # sg-7d607e19"
        "us-west-2c": dict(
            SubnetID="subnet-6217573b", Groups=["sg-d9559ca1"]), # sg-7d607e19
        "us-west-2b": dict(
            SubnetID="subnet-57253232", Groups=["sg-d9559ca1"]), # sg-7d607e19
    },
}
subnet_info = all_subnet_info[config.BUCKET]
