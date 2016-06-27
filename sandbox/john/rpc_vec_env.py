import tempfile, cPickle, subprocess, sys, numpy as np, os
import zerorpc
from vec_env import VecEnv

class ServerSideEnvWrapper(object):
    def __init__(self, envs, max_path_length):
        self.envs = envs
        self.max_path_length = max_path_length
        self.ts = np.zeros(len(envs), dtype='int')
    def vstep(self, acs_str):
        acs = loads(acs_str)
        assert len(acs) == len(self.envs)
        results = [env.step(ac) for (env, ac) in zip(self.envs, acs)]
        obs, rews, dones, _ = map(np.array, zip(*results))
        self.ts += 1
        dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done: 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        out = np.array(obs), np.array(rews), np.array(dones)
        return dumps(out)
    def vreset(self):
        result = [env.reset() for env in self.envs]
        return dumps(result)

def _start_server():
    fname = sys.argv[1]
    addr = sys.argv[2]
    k = int(sys.argv[3])
    max_path_length = int(sys.argv[4])
    with open(fname, 'r') as fh:
        s = fh.read()    
    envs = [loads(s) for _ in xrange(k)]
    server = zerorpc.Server(ServerSideEnvWrapper(envs, max_path_length))
    server.bind(addr)
    server.run()

def loads(s):
    return cPickle.loads(s)

def dumps(o):
    return cPickle.dumps(o, protocol=-1)

class EnvProxy(object):
    def __init__(self, env, i, k, max_path_length):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(dumps(env))
        f.close()
        pid = os.getpid()
        addr = "ipc:///tmp/%i_%0.2i.ipc"%(pid, i)
        self.popen = subprocess.Popen(["python", "-m", "sandbox.john.rpc_vec_env", f.name, addr, str(k), str(max_path_length)])
        self.client = zerorpc.Client()
        self.client.connect(addr)
    def __del__(self):
        self.popen.terminate()
    def vstep(self, acs):
        return self.client("vstep", dumps(acs), async=True)
    def vreset(self):
        return self.client("vreset", async=True)

class RpcVecEnv(VecEnv):
    def __init__(self, env, n, k, max_path_length = 999999):
        """
        env : original Env
        n : number of processes
        k : number of environments per process
        """
        self.remotes = [EnvProxy(env, i, k, max_path_length) for i in xrange(n)]
        self.k = k
        self._action_space = env.action_space
        self._observation_space = env.observation_space        
    def step(self, action_n):
        results = get_values([remote.vstep(action_n[i * self.k : (i+1) * self.k])
            for (i,remote) in enumerate(self.remotes)])
        obs, rews, dones = zip(*results)
        return np.concatenate(obs), np.concatenate(rews), np.concatenate(dones)
    def reset(self):        
        results = get_values([remote.vreset() for remote in self.remotes])
        return np.concatenate(results)
    @property
    def num_envs(self):
        return len(self.remotes) * self.k
    @property
    def action_space(self):
        return self._action_space
    @property
    def observation_space(self):
        return self._observation_space
    
    

def get_values(li):
    return [loads(el.get()) for el in li]

if __name__ == "__main__":
    _start_server()