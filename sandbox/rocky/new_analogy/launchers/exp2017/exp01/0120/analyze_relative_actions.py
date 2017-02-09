import pickle

from rllab.misc import logger
from sandbox.rocky.s3 import resource_manager
import matplotlib.pyplot as plt

logger.log("Loading data...")
file_name = resource_manager.get_file("fetch_relative/1000_trajs.pkl")
with open(file_name, "rb") as f:
    paths = pickle.load(f)
logger.log("Loaded")


import ipdb; ipdb.set_trace()
