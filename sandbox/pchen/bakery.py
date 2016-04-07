import argparse
from collections import defaultdict
from functools import partial

import joblib

from rllab.algos.base import Algorithm
import cPickle as pickle

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.core.parameterized import suppress_params_loading
import lasagne.objectives as LO
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI
import lasagne
import theano.tensor as TT
import numpy as np
import rllab.plotter as plotter
import rllab.misc.logger as logger
from rllab.misc.ext import stdize
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.optimizers.hessian_free_optimizer import HessianFreeOptimizer
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer


class Bakery(Algorithm, LasagnePowered):

    def __init__(
            self,
            env,
            policy,
            paths,
            optimizer,
            eval_optimizer,
            ul_obj="passive_dynamics",
            data_size=None,
            train_portion=0.8,
            eval="expert_mimic",
            test_paths=False,
            **kwargs
    ):
        old_pi = policy
        with suppress_params_loading():
            new_pi = pickle.loads(pickle.dumps(policy))

        layers = new_pi._mean_network.layers
        assert len(layers) == (1+3+1)
        last_encoder_layer = layers[2]
        bake_layers = []
        predictions_tgt_var = TT.matrix("predictions")
        input_vars = [new_pi._mean_network.input_var, predictions_tgt_var]
        if ul_obj == "passive_dynamics":
            bake_layers.append(
                L.DenseLayer(
                    incoming=last_encoder_layer,
                    num_units=50,
                )
            )
            bake_layers.append(
                L.DenseLayer(
                    incoming=bake_layers[-1],
                    num_units=env.spec.observation_space.flat_dim,
                )
            )
        elif ul_obj == "active_dynamics":
            action_var = TT.matrix("actions")
            input_vars.append(action_var)
            bake_layers.append(
                L.concat(incomings=[
                    last_encoder_layer,
                    L.InputLayer(
                        (None, env.spec.action_space.flat_dim),
                        input_var=action_var,
                    )
                ])
            )
            bake_layers.append(
                L.DenseLayer(
                    incoming=bake_layers[-1],
                    num_units=50,
                )
            )
            bake_layers.append(
                L.DenseLayer(
                    incoming=bake_layers[-1],
                    num_units=env.spec.observation_space.flat_dim,
                )
            )
        elif ul_obj == "baseline":
            bake_layers.append(
                L.DenseLayer(
                    incoming=last_encoder_layer,
                    num_units=50,
                )
            )
            bake_layers.append(
                L.DenseLayer(
                    incoming=bake_layers[-1],
                    num_units=1,
                )
            )
        elif ul_obj == "nop":
            pass
        else:
            raise "unsupported objective"
        if ul_obj != "nop":
            predictions = L.get_output(bake_layers[-1])
            loss = TT.mean(LO.squared_error(predictions, predictions_tgt_var))

            LasagnePowered.__init__(self, [bake_layers[-1]])
            optimizer.update_opt(loss, self, input_vars, network_outputs=predictions)

        info = locals()
        info.pop("self")
        self.bake(**info)

        expert_actions_var = TT.matrix("expert_actions")
        input_vars = [new_pi._mean_network.input_var, expert_actions_var]
        loss = TT.mean(
            LO.squared_error(new_pi._mean_var, expert_actions_var)
        ) + 0*TT.mean(new_pi._log_std_var)
        eval_optimizer.update_opt(loss, new_pi, input_vars, network_outputs=new_pi._mean_network.output)
        self.eval(**info)

    def eval(
            self,
            paths,
            ul_obj,
            eval_optimizer,
            input_vars,
            train_portion,
            **kwargs
    ):
        optimizer = eval_optimizer
        data = [[] for _ in input_vars]
        for path in paths:
            data[0].append(path["observations"][:-1])
            data[1].append(path["agent_infos"]["mean"][:-1])

        data = [
            (np.concatenate(ins)) for ins in data
            ]
        data[0] = stdize(data[0])
        data[1] = stdize(data[1])
        ds = data[0].shape[0]
        shuffled_idx = np.arange(ds)
        train_idx = shuffled_idx[:int(train_portion*ds)]
        test_idx = shuffled_idx[int(train_portion*ds):]
        train_data = [
            inp[train_idx] for inp in data
            ]
        test_data = [
            inp[test_idx] for inp in data
            ]

        loss = optimizer._opt_fun["f_loss"](*train_data)
        logger.log("initial train loss %s" % loss)
        def print_loss(loss, elapsed, itr, **kwargs):
            logger.log("eval itr %s ..." % itr)
            logger.log("train loss %s" % loss)
            test_loss = optimizer._opt_fun["f_loss"](*test_data)
            logger.log("test loss %s" % test_loss)
        optimizer._callback = lambda params: print_loss(**params)
        optimizer.optimize(
            train_data,
            # callback=print_loss,
        )

    def bake(
            self,
            paths,
            ul_obj,
            optimizer,
            input_vars,
            train_portion,
            **kwargs
    ):
        if ul_obj == "nop":
            return

        data = [[] for _ in input_vars]
        for path in paths:
            data[0].append(path["observations"][:-1])
            if ul_obj in ["passive_dynamics", "active_dynamics"]:
                data[1].append(path["observations"][1:])
            elif ul_obj == "baseline":
                data[1].append(path["returns"][:-1])
            if ul_obj in ["active_dynamics"]:
                data[2].append(path["actions"][:-1])

        data = [
            (np.concatenate(ins)) for ins in data
            ]
        data[0] = stdize(data[0])
        data[1] = stdize(data[1])
        ds = data[0].shape[0]
        shuffled_idx = np.arange(ds)
        train_idx = shuffled_idx[:int(train_portion*ds)]
        test_idx = shuffled_idx[int(train_portion*ds):]
        train_data = [
            inp[train_idx] for inp in data
            ]
        test_data = [
            inp[test_idx] for inp in data
            ]

        loss = optimizer._opt_fun["f_loss"](*train_data)
        logger.log("initial train loss %s" % loss)
        def print_loss(loss, elapsed, itr, **kwargs):
            logger.log("bake itr %s ..." % itr)
            logger.log("train loss %s" % loss)
            test_loss = optimizer._opt_fun["f_loss"](*test_data)
            logger.log("test loss %s" % test_loss)
        optimizer._callback = lambda params: print_loss(**params)
        optimizer.optimize(
            train_data,
            # callback=print_loss,
        )


    def dummy(self):
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']
    paths = data.get('paths', None)
    Bakery(
        policy=policy,
        env=env,
        paths=paths,
        ul_obj="passive_dynamics",
        test_paths=True,
        # ul_obj="active_dynamics",
        # ul_obj="baseline",
        # ul_obj="nop",
        # optimizer=LbfgsOptimizer(
        #     max_opt_itr=2000,
        # )
        # optimizer=HessianFreeOptimizer(
        #     max_opt_itr=300,
        # )
        optimizer=FirstOrderOptimizer(
            update_method=partial(lasagne.updates.adam, learning_rate=1e-3),
            # update_method=partial(lasagne.updates.adadelta),
            max_epochs=10,
        ),
        eval_optimizer=FirstOrderOptimizer(
            update_method=partial(lasagne.updates.adam, learning_rate=1e-4),
            # update_method=partial(lasagne.updates.adadelta),
            max_epochs=50,
        )
    )

