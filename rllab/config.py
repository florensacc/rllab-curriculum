import os.path as osp

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

LOG_DIR = PROJECT_PATH + "/data"

LOCAL_DOCKER_IMAGE = "quay.io/openai/rocky-sandbox"

OPENAI_KUBE_DOCKER_IMAGE = "quay.io/openai/rocky-sandbox"

KUBE_PREFIX = "rocky_"

DOCKER_LOG_DIR = "/tmp/expt"

S3_PATH = "s3://openai-sci-rocky/experiments"

POD_DIR = PROJECT_PATH + "/.pods"
