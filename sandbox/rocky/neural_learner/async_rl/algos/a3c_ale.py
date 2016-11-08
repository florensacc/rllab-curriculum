from rllab.core.serializable import Serializable
from ..algos.async_algo import AsyncAlgo


class PolicyWrapper(Serializable):
    def __init__(self, agent):
        Serializable.quick_init(self, locals())
        self.agent = agent#.process_copy()

    def reset(self):
        self.agent.phase = "Test"
        self.agent.act(state=None, reward=0, is_state_terminal=True)

    def get_action(self, obs):
        self.agent.phase = "Test"
        return self.agent.act(state=obs, reward=0, is_state_terminal=False)


class A3CALE(AsyncAlgo):
    def set_process_params(self, process_id, env, agent, training_args):
        agent.process_id = process_id

    def get_snapshot(self, env, agent):
        return dict(
            # algo=self,
            env=env,
            policy=PolicyWrapper(agent),
        )
