import os
from sandbox.rein.dynamics_models.bnn.run_conv_bnn import Experiment
from rllab.misc.instrument import stub, run_experiment_lite


os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

e = Experiment()

run_experiment_lite(
    e.main(),
    exp_prefix="conv_bnn_a",
    mode="lab_kube",
    dry=False,
    use_gpu=False,
    script="sandbox/rein/experiments/run_experiment_lite.py",
)
