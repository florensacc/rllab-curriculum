import tensorflow as tf



class JointNetwork(object):
    def __init__(self, env_spec):
        import ipdb; ipdb.set_trace()
        pass

    def as_summary_network(self):
        return SummaryNetwork()

    def as_action_network(self):
        return ActionNetwork()


class SummaryNetwork(object):
    def __init__(self):
        pass


class ActionNetwork(object):
    def __init__(self):
        pass


class Net(object):
    def __init__(self, obs_type):
        assert obs_type == "full_state"
        self.obs_type = obs_type

    def new_networks(self, env_spec):
        joint_network = JointNetwork(env_spec)
        summary_network = joint_network.as_summary_network()
        action_network = joint_network.as_action_network()
        return summary_network, action_network
