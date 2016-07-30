class BonusEvaluator(object):
    def __init__(self):
        pass

    def evaluate(self,states,actions,next_states):
        raise NotImplementedError

    def update(self,states,actions):
        raise NotImplementedError
