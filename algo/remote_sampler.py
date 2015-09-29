import zmq
import cloudpickle
import subprocess
import os
import pickle


class RemoteSampler(object):

    def __init__(self, sampler_module, n_parallel, gen_mdp, gen_policy):
        self._sampler_module = sampler_module
        self._n_parallel = n_parallel
        self._gen_mdp = gen_mdp
        self._gen_policy = gen_policy
        self._socket = None
        self._sampler_process = None

    def __enter__(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        port = socket.bind_to_random_port("tcp://*")
        sampler_process = subprocess.Popen(
            ['python', '-m', self._sampler_module, '-p', str(port)],
            env=dict(os.environ, THEANO_FLAGS="device=cpu")
        )
        socket.recv()
        socket.send(cloudpickle.dumps((
            self._n_parallel, self._gen_mdp, self._gen_policy)))
        socket.recv()

        self._context = context
        self._socket = socket
        self._sampler_process = sampler_process

        return self

    def request_samples(
            self, itr, cur_params, max_samples_per_itr,
            max_steps_per_itr, discount):
        self._socket.send(cloudpickle.dumps((
            itr, cur_params, max_samples_per_itr, max_steps_per_itr, discount
        )))
        self._socket.recv()
        return pickle.loads(self._socket.recv())

    def __exit__(self, exc_type, exc_value, traceback):
        if self._sampler_process:
            self._sampler_process.terminate()
            self._sampler_process = None
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None
