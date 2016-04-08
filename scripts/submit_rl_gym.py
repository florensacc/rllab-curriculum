from __future__ import print_function
from __future__ import absolute_import
import argparse
import joblib
from rllab.envs.rl_gym_env import RLGymEnv
from rl_gym import scoreboard


class PolicyAgentWrapper(object):
    def __init__(self, policy, algo):
        self.policy = policy
        self.name = "{algo}_{policy}".format(algo=algo.__class__.__name__, policy=policy.__class__.__name__)

    def act(self, ob):
        return self.policy.get_action(ob)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()
    data = joblib.load(args.file)
    policy = data["policy"]
    env = data["env"]
    algo = data["algo"]
    assert isinstance(env, RLGymEnv)
    agent = PolicyAgentWrapper(policy, algo)
    training_episode_batch = scoreboard.upload_training_episode_batch(
        episode_rewards=data["episode_rewards"].tolist(),
        episode_lengths=data["episode_lengths"].tolist()
    )
    evaluation = scoreboard.evaluate(
        agent_callable=lambda ob_n, _, __: [agent.act(ob) for ob in ob_n],
        agent_name=agent.name,
        env_id=env.env_id,
        training_episode_batch_id=training_episode_batch.id)
