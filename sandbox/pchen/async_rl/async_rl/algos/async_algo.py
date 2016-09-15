import argparse
import copy
import multiprocessing as mp
import os
import sys
import statistics
import time
import pickle
sys.path.append('.')

import numpy as np
import logging
from rllab.misc import logger, ext
from sandbox.pchen.async_rl.async_rl.utils.random_seed import set_random_seed
from sandbox.pchen.async_rl.async_rl.utils.picklable import Picklable


class AsyncAlgo(Picklable):
    def __init__(self,
        n_processes,
        env,
        agent,
        seeds=None,
        profile=False,
        logging_level=logging.INFO,
        total_steps=10**6,
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
        # we will assume the env doesn't share anything
        # self.env.prepare_sharing()
        self.agent.prepare_sharing()

        if self.profile:
            train_func = self.train_one_process_with_profile
        else:
            train_func = self.train_one_process

        if self.n_processes == 1:
            train_func(0, global_vars, self.training_args)
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

    def child_process_setup(self, process_id, global_vars, args):
        logger.log("Process %d: setup."%(process_id),color="yellow")

        # Set seed
        seed = self.seeds[process_id]
        ext.set_seed(seed)

    def child_fresh_env_agent(self):
        # Copy from the mother env, agent
        env = pickle.loads(pickle.dumps(self.env))
        agent = self.agent.process_copy()
        agent.phase = "Train"
        return (env, agent)


    def train_one_process(self, process_id, global_vars, args):
        """
        Set seed
        Initialize env, agent, model, opt from the base ones.
        Let shared params point to shared memory.
        Set process-specific parameters.
        """

        self.child_process_setup(process_id, global_vars, args)
        env, agent = self.child_fresh_env_agent()

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

            obs, reward, terminal, extra = env.reset(), 0., False, {}

            # each following loop is one time step

            start_time = time.time()
            while global_t < args["total_steps"]:
                # Training ------------------------------------------------------
                # Update time step counters
                with global_vars["global_t"].get_lock():
                    global_vars["global_t"].value += 1
                    global_t = global_vars["global_t"].value

                agent.update_params(
                    global_vars=global_vars,
                    training_args=args,
                )

                local_t += 1
                episode_t += 1

                # Update reward stats
                # episode_r += env.reward

                # Update agent, env
                # take actions
                # action = agent.act(
                #     env.state, env.reward, env.is_terminal, env.extra_infos,
                #     global_vars=global_vars,
                #     training_args=args,
                # )
                action = agent.act(
                    obs, reward, terminal, extra,
                    global_vars=global_vars,
                )
                obs, reward, terminal, extra = env.step(action)

                episode_r += reward

                if terminal or (episode_t > args["horizon"]):
                    # log info for each episode
                    if process_id == 0:
                        logger.log('global_t:{} local_t:{} episode_t:{} episode_r:{} t_per_sec:{}'.format(
                            global_t, local_t, episode_t, episode_r, global_t / (time.time()-start_time)))
                    episode_r = 0
                    episode_t = 0
                    obs, reward, terminal, extra = env.reset(), 0., False, {}

                    if episode_t > args["horizon"]:
                        logger.log(
                            "WARNING: horizon %d exceeded."%(args["horizon"]),
                            color="yellow",
                        )

                self.epoch = global_t // args["eval_frequency"]
                test_t = self.epoch * args["eval_frequency"]

                # Testing  -------------------------------------------------------
                # only the process that hits exactly the evaluation time will do testing
                if global_t == test_t:
                    scores = self.evaluate_performance(
                        n_runs=args["eval_n_runs"],
                        horizon=args["eval_horizon"]
                    )

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

    def evaluate_performance(self, n_runs, horizon):
        env, agent = self.child_fresh_env_agent()

        logger.log("Evaluating test performance",color="yellow")
        agent.phase = "Test"

        scores = np.zeros(n_runs)
        for i in range(n_runs):
            obs, reward, terminal, extra = env.reset(), 0., False, {}
            t = 0
            while not terminal:
                action = agent.act(obs, reward, terminal, extra)
                obs, reward, terminal, extra = env.step(action)
                scores[i] += reward
                t += 1
                if t > horizon:
                    logger.log(
                        "WARNING: test horizon %d exceeded."%(horizon),
                        color="yellow",
                    )
                    break
            logger.log(
                "Finished testing #%d with score %f."%(i,scores[i]),
                color="green",
            )
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

