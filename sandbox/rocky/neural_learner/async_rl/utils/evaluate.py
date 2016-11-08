from rllab.misc import logger
import numpy as np

def evaluate_performance(env,agent,n_runs,horizon):
    logger.log("Evaluating test performance",color="yellow")
    env.phase = "Test"
    agent.phase = "Test"

    scores = np.zeros(n_runs)
    for i in range(n_runs):
        env.initialize()
        t = 0
        while not env.is_terminal:
            action, _ = agent.act(env.state, env.reward, env.is_terminal, env.extra_infos)
            reward = env.receive_action(action)
            scores[i] += reward
            t += 1
            if t > horizon:
                logger.log(
                    "WARNING: test horizon %d exceeded."%(horizon),
                    color="yellow",
                )
                break
        logger.log(
            "Finished testing #%d with score %f."%(i,scores[i]),
            color="green",
        )
    return scores
