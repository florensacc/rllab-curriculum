from sandbox.sandy.async_rl.algos.async_algo import AsyncAlgo
from rllab.misc.overrides import overrides

class DQNALE(AsyncAlgo):
    def set_process_params(self,process_id,env,agent,training_args):
        """
        TODO: set different eps_end for different processes
        """
        self.process_id = process_id
        agent.process_id = process_id


    def get_snapshot(self,env,agent):
        return self
