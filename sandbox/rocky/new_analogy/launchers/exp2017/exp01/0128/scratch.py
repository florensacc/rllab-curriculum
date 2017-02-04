import joblib

from sandbox.rocky.new_analogy.exp_utils import run_local, run_local_docker
import numpy as np

from sandbox.rocky.s3 import resource_manager


def run_task(*_):
    from sandbox.rocky.new_analogy import fetch_utils
    n = 1000
    for height in [2, 3, 4, 5]:
        xinits = fetch_utils.collect_xinits(height=height, seeds=np.arange(n))
        file_name = resource_manager.tmp_file_name("pkl")
        joblib.dump(xinits, file_name, compress=3)
        resource_manager.register_file("fetch_{n}_xinits_{k}_boxes.pkl".format(n=n, k=height), file_name)

run_local_docker(
    run_task,
    # n_parallel=0,
    use_gpu=False,
)
