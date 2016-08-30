# url to the docker image. no need to change
DOCKER_IMAGE = "dementrock/rllab-shared"

# where to store log inside docker. no need to change
DOCKER_LOG_DIR = "/tmp/expt"

# change this to s3://$YOUR_BUCKET/experiments
AWS_S3_PATH = "s3://rocky-rllab-data-trevor-aws/davis-experiments"

# change this to s3://$YOUR_BUCKET/code
AWS_CODE_SYNC_S3_PATH = "s3://rocky-rllab-data-trevor-aws/davis-code"

# no need to change
AWS_IMAGE_ID = "ami-6a3c440a"

# change this to the instance type you'd like to change
AWS_INSTANCE_TYPE = "m4.large"

# change this to the name of the key pair you created in step 2
AWS_KEY_NAME = "davis_north_cal"

# whether to use spot instance. should almost always be True to save cost
AWS_SPOT = True

# Maximum spot price. This will depend on the instance type you select.
AWS_SPOT_PRICE = '0.5'

# Set this to your access key value. You can either replace it with the actual string here, or set the environment variable (e.g. in your ~/.bashrc file)
AWS_ACCESS_KEY = "AKIAIOSTO5ENMT3YBWGQ"

# Set this to your access secret value
AWS_ACCESS_SECRET = "OVYgDkDKQIMze5QeezrXUYSoAOHnNwP8KjqXR9tX"

# no need to change
AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

# no need to change
AWS_SECURITY_GROUPS = []

# no need to change
AWS_SECURITY_GROUP_IDS = []

# no need to change
AWS_NETWORK_INTERFACES = [
    dict(
        SubnetId="subnet-8fbfe4d6",
        Groups=["sg-00d95f64"],
        DeviceIndex=0,
        AssociatePublicIpAddress=True,
    ),
]

# configure the region for running the experiments. can use the default for now, but sometimes spot instances in
# other regions have cheaper price
AWS_REGION_NAME = "us-west-1"

# configure the files that will be ignored when syncing code. can use the default for now
CODE_SYNC_IGNORES = ["*.git/*", "*data/*", "*src/*",
                     "*.pods/*", "*tests/*", "*examples/*", "docs/*", "*.idea/*", ".DS_Store",
                     ".ipynb_checkpoints/*",
                     "blackbox/*", "blackbox.zip", "*.pyc", "*.ipynb", "*scratch-notebooks/*"]
