import os
from sandbox.rein.dynamics_models.pixel_cnn.train_double_cnn import ExperimentPixelCNN
from rllab.misc.instrument import stub, run_experiment_lite

os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

e = ExperimentPixelCNN()

run_experiment_lite(
    e.main(),
    exp_prefix="pixelcnn_a",
    mode="lab_kube",
    dry=False,
    use_gpu=True,
    script="sandbox/rein/experiments/run_experiment_lite.py",
)
