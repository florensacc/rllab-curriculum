import argparse
import types
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
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler import parallel_sampler


class Bakery(Algorithm, LasagnePowered):

    def __init__(
            self,
            env,
            policy,
            new_pi,
            paths,
            optimizer,
            eval_optimizer,
            data_size=None,
            ul_obj="passive_dynamics",
            train_portion=0.8,
            eval="expert_mimic",
            test_paths=False,
            fixed_encoder=False,
            max_path_length=100,
            whole_paths=True,
            bake_hidden_sizes=(50, 25),
            **kwargs
    ):
        old_pi = policy
        # with suppress_params_loading():
        #     new_pi = pickle.loads(pickle.dumps(policy))

        cur_ds = sum(p["observations"].shape[0] for p in paths)
        print "cur data size", cur_ds
        if data_size and data_size > cur_ds:
            diff_ds = data_size - cur_ds
            cur_params = policy.get_param_values()
            parallel_sampler.populate_task(env, policy)
            parallel_sampler.request_samples(
                policy_params=cur_params,
                max_samples=diff_ds,
                max_path_length=max_path_length,
                whole_paths=whole_paths,
            )
            paths = paths + parallel_sampler.collect_paths()

        print "env", env

        layers = new_pi._mean_network.layers
        print "pi has ", len(layers)
        last_encoder_layer = layers[2]
        bake_layers = [last_encoder_layer]
        predictions_tgt_var = TT.matrix("predictions")
        input_vars = [new_pi._mean_network.input_var, predictions_tgt_var]
        if ul_obj in ["passive_dynamics", "ae"]:
            for size in bake_hidden_sizes:
                bake_layers.append(
                    L.DenseLayer(
                        incoming=bake_layers[-1],
                        num_units=size,
                    )
                )
            bake_layers.append(
                L.DenseLayer(
                    incoming=bake_layers[-1],
                    num_units=env.spec.observation_space.flat_dim,
                    nonlinearity=None,
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
            for size in bake_hidden_sizes:
                bake_layers.append(
                    L.DenseLayer(
                        incoming=bake_layers[-1],
                        num_units=size,
                    )
                )
            bake_layers.append(
                L.DenseLayer(
                    incoming=bake_layers[-1],
                    num_units=env.spec.observation_space.flat_dim,
                    nonlinearity=None,
                )
            )
        elif ul_obj == "baseline":
            for size in bake_hidden_sizes:
                bake_layers.append(
                    L.DenseLayer(
                        incoming=bake_layers[-1],
                        num_units=size,
                    )
                )
            bake_layers.append(
                L.DenseLayer(
                    incoming=bake_layers[-1],
                    num_units=1,
                    nonlinearity=None,
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

        if fixed_encoder:
            def get_params(self, **tags):
                params = super(type(self), self).get_params(**tags)
                # print params
                # print "returning", params[4:]
                return params[4:]
            new_pi.get_params = types.MethodType(get_params, new_pi)
        expert_actions_var = TT.matrix("expert_actions")
        input_vars = [new_pi._mean_network.input_var, expert_actions_var]
        loss = TT.mean(
            LO.squared_error(new_pi._mean_var, expert_actions_var)
        ) + 0*TT.mean(new_pi._log_std_var)
        eval_optimizer.update_opt(loss, new_pi, input_vars, network_outputs=new_pi._mean_network.output)

        info = locals()
        info.pop("self")
        self.eval(**info)

    def eval(
            self,
            paths,
            ul_obj,
            eval_optimizer,
            input_vars,
            train_portion,
            test_paths,
            **kwargs
    ):
        optimizer = eval_optimizer
        def prep_data(paths):
            data = [[] for _ in input_vars]
            for path in paths:
                data[0].append(path["observations"][:-1])
                data[1].append(path["agent_infos"]["mean"][:-1])
            data = [
                (np.concatenate(ins)) for ins in data
            ]
            data[0] = stdize(data[0])
            data[1] = stdize(data[1])
            return data

        if test_paths:
            cutoff = int(len(paths) * (train_portion if test_paths else 1))
            train_data = prep_data(paths[:cutoff])
            test_data = prep_data(paths[cutoff:])
        else:
            data = prep_data(paths)
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
            # logger.log("eval itr %s ..." % itr)
            # logger.log("train loss %s" % loss)
            test_loss = optimizer._opt_fun["f_loss"](*test_data)
            # logger.log("test loss %s" % test_loss)
            logger.push_prefix('eval itr #%d | ' % itr)
            logger.record_tabular('TrainLoss', loss)
            logger.record_tabular('EvalLoss', test_loss)
            logger.dump_tabular(with_prefix=True)
            logger.pop_prefix()
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
            test_paths,
            predictions,
            **kwargs
    ):
        if ul_obj == "nop":
            return

        def prep_data(paths):
            data = [[] for _ in input_vars]
            for path in paths:
                data[0].append(path["observations"][:-1])
                if ul_obj in ["passive_dynamics", "active_dynamics"]:
                    data[1].append(path["observations"][1:])
                elif ul_obj == "baseline":
                    data[1].append(path["returns"][:-1])
                elif ul_obj == "ae":
                    data[1].append(path["observations"][:-1])
                if ul_obj in ["active_dynamics"]:
                    data[2].append(path["actions"][:-1])
            data = [
                (np.concatenate(ins)) for ins in data
                ]
            data[0] = stdize(data[0])
            data[1] = stdize(data[1])
            return data

        if test_paths:
            cutoff = int(len(paths) * (train_portion if test_paths else 1))
            train_data = prep_data(paths[:cutoff])
            test_data = prep_data(paths[cutoff:])
        else:
            data = prep_data(paths)
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
            # logger.log("train loss %s" % loss)
            # logger.log("%s" % predictions.eval({input_vars[0]: train_data[0]}))
            test_loss = optimizer._opt_fun["f_loss"](*test_data)
            logger.push_prefix('bake itr #%d | ' % itr)
            logger.record_tabular('TrainLoss', loss)
            logger.record_tabular('EvalLoss', test_loss)
            logger.dump_tabular(with_prefix=True)
            logger.pop_prefix()
        optimizer._callback = lambda params: print_loss(**params)

        optimizer.optimize(
            train_data,
            # callback=print_loss,
        )
        test_loss = optimizer._opt_fun["f_loss"](*test_data)
        logger.log("test loss %s" % test_loss)


    def dummy(self):
        pass

class Loader(object):
    def __init__(
            self,
            file,
            **kwargs
    ):
        data = joblib.load(file)
        policy = data['policy']
        env = data['env']
        paths = data.get('paths', None)
        Bakery(
            policy=policy,
            env=env,
            paths=paths,
            new_pi=GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(100, 50, 50, 25, )
            ),
            **kwargs
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

    parallel_sampler.config_parallel_sampler(3, 42)
    Bakery(
        policy=policy,
        new_pi=GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(40, 20, 20,)
        ),
        env=env,
        paths=paths,
        test_paths=True,
        fixed_encoder=True,
        bake_hidden_sizes=(10,),
        # ul_obj="ae",
        # ul_obj="passive_dynamics",
        # ul_obj="active_dynamics",
        # ul_obj="baseline",
        ul_obj="nop",
        # data_size=100000,
        # optimizer=LbfgsOptimizer(
        #     max_opt_itr=10,
        # ),
        # optimizer=HessianFreeOptimizer(
        #     max_opt_itr=10,
        # ),
        optimizer=FirstOrderOptimizer(
            update_method=partial(lasagne.updates.adam, learning_rate=1e-3),
            # update_method=partial(lasagne.updates.adadelta),
            max_epochs=50,
        ),
        # eval_optimizer=LbfgsOptimizer(
        #     max_opt_itr=500,
        # ),
        eval_optimizer=FirstOrderOptimizer(
            update_method=partial(lasagne.updates.adam, learning_rate=1e-3),
            # update_method=partial(lasagne.updates.adadelta),
            max_epochs=500,
        )
    )

