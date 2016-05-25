from __future__ import print_function
from __future__ import absolute_import
from rllab.misc.instrument import run_experiment_lite as theano_run_experiment_lite


def run_experiment_lite(*args, **kwargs):
    return theano_run_experiment_lite(script="sandbox/rocky/tf/scripts/run_experiment_lite.py", *args, **kwargs)
