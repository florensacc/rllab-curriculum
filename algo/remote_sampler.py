import zmq
import cloudpickle
import subprocess
import os
import pickle
import pydoc


class RemoteSampler(object):

    def __init__(self, sampler_module, n_parallel, gen_mdp, gen_policy, savedir):
        self.sampler_module = sampler_module
        self.n_parallel = n_parallel
        self.gen_mdp = gen_mdp
        self.gen_policy = gen_policy
        self.socket = None
        self.sampler_process = None
        self.context = None
        self.socket = None
        self.sampler = None
        self.savedir = savedir

    def __enter__(self):
        if self.n_parallel > 1:
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            port = socket.bind_to_random_port("tcp://*")
            sampler_process = subprocess.Popen(
                ['python', '-m', self.sampler_module, '-p', str(port)],
                env=dict(os.environ, THEANO_FLAGS="device=cpu")
            )
            socket.recv()
            socket.send(cloudpickle.dumps((
                self.n_parallel, self.gen_mdp, self.gen_policy, self.savedir)))
            socket.recv()

            self.context = context
            self.socket = socket
            self.sampler_process = sampler_process
        else:
            self.sampler = pydoc.locate(self.sampler_module).sampler(buf=None, gen_mdp=self.gen_mdp, gen_policy=self.gen_policy, n_parallel=self.n_parallel)
        return self

    def request_samples(
            self, itr, cur_params, max_samples_per_itr,
            max_steps_per_itr, discount):
        if self.n_parallel > 1:
            self.socket.send(cloudpickle.dumps((
                itr, cur_params, max_samples_per_itr, max_steps_per_itr, discount
            )))
            return pickle.loads(self.socket.recv())
        else:
            return self.sampler.collect_samples(itr, cur_params, max_samples_per_itr, max_steps_per_itr, discount)


    def __exit__(self, exc_type, exc_value, traceback):
        if self.sampler_process:
            self.sampler_process.terminate()
            self.sampler_process = None
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None
