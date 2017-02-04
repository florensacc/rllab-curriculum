import torch
import os


def is_cuda():
    return torch.cuda.is_available() and os.environ.get("RLLAB_USE_GPU", False)
