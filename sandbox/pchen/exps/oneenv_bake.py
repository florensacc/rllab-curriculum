import os

import joblib

from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.pchen.bakery import Bakery, Loader

os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()
    file = args.file

    # data = joblib.load(args.file)
    # import ipdb; ipdb.set_trace()
    # policy = data['policy']
    # env = data['env']
    # paths = data.get('paths', None)

    b = Loader(
        file=file,
        # new_pi=gaussianmlppolicy(
        #     env_spec=env.spec,
        #     hidden_sizes=(100, 50, 50, 30, 25, )
        # ),
        test_paths=True,
        fixed_encoder=True,
        bake_hidden_sizes=(50, 30, 25,),
        ul_obj="passive_dynamics",
        # ul_obj="active_dynamics",
        # ul_obj="baseline",
        # ul_obj="nop",
        # optimizer=LbfgsOptimizer(
        #     max_opt_itr=2000,
        # )
        # optimizer=HessianFreeOptimizer(
        #     max_opt_itr=300,
        # )
        # data_size=100000,
        optimizer=FirstOrderOptimizer(
            # update_method=partial(lasagne.updates.adadelta),
            max_epochs=10,
        ),
        eval_optimizer=LbfgsOptimizer(
            max_opt_itr=500,
        ),
        # eval_optimizer=FirstOrderOptimizer(
        #     update_method=partial(lasagne.updates.adam, learning_rate=1e-4),
        #     # update_method=partial(lasagne.updates.adadelta),
        #     max_epochs=500,
        # )
    )
    run_experiment_lite(
        b.dummy(),
        exp_prefix="bakery",
        n_parallel=2,
        snapshot_mode="all",
        seed=2,
        mode="local",
        # mode="lab_kube",
        # resouces=dict(
        #     requests=dict(
        #         cpu=3.4
        #     )
        # )
    )
    b = Loader(
        file=file,
        # new_pi=gaussianmlppolicy(
        #     env_spec=env.spec,
        #     hidden_sizes=(100, 50, 50, 30, 25, )
        # ),
        test_paths=True,
        fixed_encoder=True,
        bake_hidden_sizes=(50, ),
        ul_obj="passive_dynamics",
        # ul_obj="active_dynamics",
        # ul_obj="baseline",
        # ul_obj="nop",
        # optimizer=LbfgsOptimizer(
        #     max_opt_itr=2000,
        # )
        # optimizer=HessianFreeOptimizer(
        #     max_opt_itr=300,
        # )
        # data_size=100000,
        optimizer=FirstOrderOptimizer(
            # update_method=partial(lasagne.updates.adadelta),
            max_epochs=10,
        ),
        eval_optimizer=LbfgsOptimizer(
            max_opt_itr=500,
        ),
        # eval_optimizer=FirstOrderOptimizer(
        #     update_method=partial(lasagne.updates.adam, learning_rate=1e-4),
        #     # update_method=partial(lasagne.updates.adadelta),
        #     max_epochs=500,
        # )
    )
    run_experiment_lite(
        b.dummy(),
        exp_prefix="bakery",
        n_parallel=2,
        snapshot_mode="all",
        seed=2,
        mode="local",
        # mode="lab_kube",
        # resouces=dict(
        #     requests=dict(
        #         cpu=3.4
        #     )
        # )
    )
    b = Loader(
        file=file,
        # new_pi=gaussianmlppolicy(
        #     env_spec=env.spec,
        #     hidden_sizes=(100, 50, 50, 30, 25, )
        # ),
        test_paths=True,
        fixed_encoder=True,
        bake_hidden_sizes=(50, 25,),
        ul_obj="passive_dynamics",
        # ul_obj="active_dynamics",
        # ul_obj="baseline",
        # ul_obj="nop",
        # optimizer=LbfgsOptimizer(
        #     max_opt_itr=2000,
        # )
        # optimizer=HessianFreeOptimizer(
        #     max_opt_itr=300,
        # )
        # data_size=100000,
        optimizer=FirstOrderOptimizer(
            # update_method=partial(lasagne.updates.adadelta),
            max_epochs=10,
        ),
        eval_optimizer=LbfgsOptimizer(
            max_opt_itr=500,
        ),
        # eval_optimizer=FirstOrderOptimizer(
        #     update_method=partial(lasagne.updates.adam, learning_rate=1e-4),
        #     # update_method=partial(lasagne.updates.adadelta),
        #     max_epochs=500,
        # )
    )
    run_experiment_lite(
        b.dummy(),
        exp_prefix="bakery",
        n_parallel=2,
        snapshot_mode="all",
        seed=2,
        mode="local",
        # mode="lab_kube",
        # resouces=dict(
        #     requests=dict(
        #         cpu=3.4
        #     )
        # )
    )
    b = Loader(
        file=file,
        # new_pi=gaussianmlppolicy(
        #     env_spec=env.spec,
        #     hidden_sizes=(100, 50, 50, 30, 25, )
        # ),
        test_paths=True,
        fixed_encoder=True,
        bake_hidden_sizes=(50, 25,),
        ul_obj="passive_dynamics",
        # ul_obj="active_dynamics",
        # ul_obj="baseline",
        # ul_obj="nop",
        # optimizer=LbfgsOptimizer(
        #     max_opt_itr=2000,
        # )
        # optimizer=HessianFreeOptimizer(
        #     max_opt_itr=300,
        # )
        # data_size=100000,
        optimizer=FirstOrderOptimizer(
            # update_method=partial(lasagne.updates.adadelta),
            max_epochs=10,
        ),
        # eval_optimizer=LbfgsOptimizer(
        #     max_opt_itr=500,
        # ),
        eval_optimizer=FirstOrderOptimizer(
            update_method=partial(lasagne.updates.adam, learning_rate=1e-4),
            # update_method=partial(lasagne.updates.adadelta),
            max_epochs=500,
        )
    )
    run_experiment_lite(
        b.dummy(),
        exp_prefix="bakery",
        n_parallel=2,
        snapshot_mode="all",
        seed=2,
        mode="local",
        # mode="lab_kube",
        # resouces=dict(
        #     requests=dict(
        #         cpu=3.4
        #     )
        # )
    )
