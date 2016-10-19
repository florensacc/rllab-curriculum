from sandbox.rocky.tf.policies.base import Policy


class AnalogyPolicy(Policy):
    def apply_demo(self, path):#demo_obs, demo_actions):
        """
        Process the given demonstration. This is usually called after resetting the policy and before applying the
        policy in the test environment.
        :param path: The demonstration path, consisting of observations, actions, etc.
        :return: This method does not need to return anything.
        """
        pass
