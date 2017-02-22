import argparse
import copy
import multiprocessing as mp
import os
import sys
import statistics
import time
sys.path.append('.')

import numpy as np
import logging
from rllab.misc import logger
from sandbox.sandy.misc.random_seed import set_random_seed
from sandbox.sandy.async_rl.utils.picklable import Picklable


class AsyncAlgo(Picklable):
    def __init__(self,
        n_processes,
        env,
        agent,
        seeds=None,
        profile=False,
        logging_level=logging.INFO,
        total_steps=8*10**7,
        eval_frequency=10**6,
        eval_n_runs=10,
        horizon=np.inf,
        eval_horizon=np.inf,
        **kwargs
    ):
        self.env = env
        self.agent = agent
        self.n_processes = n_processes
        self.seeds = seeds
        self.profile=profile
        self.logging_level=logging_level

        # The following arguments must be used by a single processor, instead of globally managed
        self.training_args = dict(
            total_steps=total_steps,
            eval_frequency=eval_frequency,
            eval_n_runs=eval_n_runs,
            horizon=horizon,
            eval_horizon=eval_horizon,
        )

        if self.seeds is None:
            self.seeds = np.random.randint(low=0, high=2 ** 32, size=(n_processes))
        self.unpicklable_list = ["env","agent","test_env","test_agent"] # only need to store cur_env and cur_agent, see train_one_process()

    def train(self):
        # Global synchronized variables controlling the training process.
        global_vars = {
            "global_t": mp.Value('l',0),
            "max_score": mp.Value('f',np.finfo(np.float32).min),
            "process_file_header_written": mp.Value('i',0),
            "start_time": time.time()
        }

        os.environ['OMP_NUM_THREADS'] = '1'
        logging.basicConfig(level=self.logging_level)
        self.env.prepare_sharing()
        self.agent.prepare_sharing()

        if self.profile:
            train_func = self.train_one_process_with_profile
        else:
            train_func = self.train_one_process

        if self.n_processes == 1:
            train_func(0,global_vars,self.training_args)
        else:
            # Train each process
            processes = []

            for process_id in range(self.n_processes):
                processes.append(
                    mp.Process(
                        target=train_func,
                        args=(process_id,global_vars,self.training_args)
                    )
                )

            for p in processes:
                p.start()

            for p in processes:
                p.join()

    def train_one_process(self, process_id, global_vars, args):
        """
        Set seed
        Initialize env, agent, model, opt from the base ones.
        Let shared params point to shared memory.
        Set process-specific parameters.
        """
        self.process_id = process_id

        # Copy from the mother env, agent
        logger.log("Process %d: copying environment and agent for training."%(process_id),color="yellow")
        env = self.env.process_copy()
        env.phase = "Train"
        agent = self.agent.process_copy()
        agent.phase = "Train"
        self.cur_env = env
        self.cur_agent = agent

        # Set seed
        seed = self.seeds[process_id]
        set_random_seed(seed)
        env.set_seed(seed)
        print("Setting process seed to", seed)

        try:
            self.prev_global_t = 0
            global_t = global_vars["global_t"].value # useful if recover from a snapshot
            local_t = 0
            episode_t = 0
            episode_r = 0

            self.set_process_params(
                process_id=process_id,
                env=env,
                agent=agent,
                training_args=args,
            )

            # each following loop is one time step
            while global_t < args["total_steps"]:
                # Training ------------------------------------------------------
                # Update time step counters
                with global_vars["global_t"].get_lock():
                    global_vars["global_t"].value += 1
                    global_t = global_vars["global_t"].value
                local_t += 1
                episode_t += 1

                # Update reward stats
                episode_r += env.reward

                # Update agent, env
                env.update_params(
                    global_vars=global_vars,
                    training_args=args,
                )
                agent.update_params(
                    global_vars=global_vars,
                    training_args=args,
                )

                # take actions
                is_terminal = env.is_terminal or (episode_t > args["horizon"])
                action = agent.act(
                    env.state, env.reward, is_terminal, env.extra_infos,
                    global_vars=global_vars,
                    training_args=args,
                )

                if is_terminal:
                    # log info for each episode
                    if process_id == 0:
                        logger.log('global_t:{} local_t:{} episode_t:{} episode_r:{}'.format(
                            global_t, local_t, episode_t, episode_r))
                    episode_r = 0
                    episode_t = 0
                    env.initialize()

                    if episode_t > args["horizon"]:
                        logger.log(
                            "WARNING: horizon %d exceeded."%(args["horizon"]),
                            color="yellow",
                        )
                else:
                    env.receive_action(action)

                self.epoch = global_t // args["eval_frequency"]
                test_t = self.epoch * args["eval_frequency"]
                # Testing  -------------------------------------------------------
                # only the agent that hits exactly the evaluation time will do testing
                if global_t == test_t:
                    # Test env may change; test recurrent agents need initialized hidden states
                    logger.log("Process %d: copying env and agent for testing."%(process_id),color="yellow")
                    self.test_env = env.process_copy()
                    # need to copy the env, in case that the current traj does not finish
                    self.test_agent = agent.process_copy()
                    # if using recurrent agents, beware to not change the hidden states

                    scores = self.evaluate_performance(
                        n_runs=args["eval_n_runs"],
                        horizon=args["eval_horizon"]
                    )
                    del self.test_env
                    del self.test_agent
                    agent.phase = "Train"

                    # Notice that only one thread can report.
                    elapsed_time = time.time() - global_vars["start_time"]
                    logger.record_tabular('Epoch',self.epoch)
                    logger.record_tabular('GlobalT',global_t)
                    logger.record_tabular('ElapsedTime',elapsed_time)
                    logger.record_tabular_misc_stat('Return',scores)

                    agent.finish_epoch(epoch=self.epoch,log=True)

                    # Update max score
                    mean = np.average(scores)
                    max_score = global_vars["max_score"]
                    with max_score.get_lock():
                        max_score_update = mean > max_score.value
                        if max_score_update:
                            max_score.value = mean
                            print('The best score is updated {} -> {}'.format(
                                max_score.value, mean))

                    # Avoid writing more than one tabular header due to multiprocessing
                    header_written = global_vars["process_file_header_written"]
                    with header_written.get_lock():
                        if header_written.value == 0:
                            logger.dump_tabular(write_header=True)
                            header_written.value = 1
                        else:
                            logger.dump_tabular(write_header=False)

                    # Save snapshots
                    params = self.get_snapshot(env,agent)
                    logger.save_itr_params(self.epoch,params)

                # Each agent not doing test will also clear its epoch stats
                if (global_t > test_t) and (self.prev_global_t < test_t):
                    agent.finish_epoch(epoch=self.epoch,log=False)
                self.prev_global_t = global_t

        except KeyboardInterrupt:
            if process_id == 0:
                self.epoch = global_t // args["eval_frequency"]
                params = self.get_snapshot(env,agent)
                logger.save_itr_params(self.epoch,params)
            raise

    def evaluate_performance(self,n_runs,horizon,return_paths=False,deterministic=False,check_equiv=False):
        # check_equiv - if true, checks that LSTM cell and hidden states are equivalent
        # between this (target) policy and adversarial policy, if there is one

        #logger.log("Process %d: evaluating test performance"%(self.process_id),color="yellow")
        logger.log("Evaluating test performance:",color="yellow")
        self.test_env.phase = "Test"
        self.test_agent.phase = "Test"
        env = self.test_env
        agent = self.test_agent

        scores = np.zeros(n_runs)
        if return_paths:
            paths = [{'rewards':[], 'states':[], 'actions':[]} for i in range(n_runs)]
        for i in range(n_runs):
            env.initialize()
            t = 0
            while not env.is_terminal:
                if return_paths:
                    paths[i]['states'].append(env.state)
                if check_equiv and env.adversary_fn is not None:
                    adv_lstm_c, adv_lstm_h = env.adversary_fn(None, None)
                
                action = agent.act(env.state, env.reward, env.is_terminal, env.extra_infos, deterministic=deterministic)
                reward = env.receive_action(action)

                if check_equiv and env.adversary_fn is not None:
                    assert np.array_equal(agent.model.lstm.h.data, adv_lstm_h), \
                            "LSTM hidden states not equal"
                    assert np.array_equal(agent.model.lstm.c.data, adv_lstm_c), \
                            "LSTM cell states not equal"
                if return_paths:
                    paths[i]['actions'].append(action)
                    paths[i]['rewards'].append(reward)

                scores[i] += reward
                t += 1
                if t > horizon:
                    logger.log(
                        "Process %d: WARNING: test horizon %d exceeded."%(self.process_id, horizon),
                        color="yellow",
                    )
                    break
            logger.log(
                "Process %d: finished testing #%d with score %f."%(self.process_id, i,scores[i]),
                color="green",
            )

        if return_paths:
            return scores, paths
        return scores

    def train_one_process_with_profile(self, process_id, global_vars, args):
        import cProfile
        cmd = 'self.train_loop(process_id, global_vars, args)'
        cProfile.runctx(
            cmd, globals(), locals(),
            os.path.join(
                logger.get_snapshot_dir(),
                'profile-{}.out'.format(os.getpid())
            ),
        )
    def set_process_params(self,process_id,env,agent,training_args):
        """
        To be implemented by subclass.
        For example, assign different end_eps to different agent.
        """
        raise NotImplementedError


    def get_snapshot(self,env,agent):
        raise NotImplementedError
