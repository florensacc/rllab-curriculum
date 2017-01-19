from sandbox.sandy.async_rl.algos.async_algo import AsyncAlgo
from rllab.misc.overrides import overrides

class A3CALE(AsyncAlgo):
    def set_process_params(self,process_id,env,agent,training_args):
        agent.process_id = process_id

    def get_snapshot(self,env,agent):
        return dict(algo=self)
