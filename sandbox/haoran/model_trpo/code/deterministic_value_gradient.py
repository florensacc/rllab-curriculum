from rllab.algos.base import RLAlgorithm
from sandbox.haoran.model_trpo.code.utils import rollout
import rllab.misc.logger as logger
import rllab.plotter as plotter
import numpy as np

class DeterministicValueGradient(RLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            n_itr=500,
            max_path_length=500,
            discount=0.99,
            plot=False,
            pause_for_plot=False,
            lr=0.01,
        ):
        self.env = env
        self.policy = policy
        self.n_itr = n_itr
        self.max_path_length = max_path_length
        self.discount = discount
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.lr = lr

    def compute_policy_gradient(self,env,policy,path,gamma):
        T = len(path["rewards"]) # path length
        states = path["env_infos"]["full_states"]
        n = len(states[0]) # state dimension
        D = len(policy.get_theta()) # parameter dimension
        V_theta = np.zeros((T,D)) # gradient wrt params at all time steps
        V_s = np.zeros((T,n)) # gradient wrt states at all time steps

        for t in range(T-2,-1,-1):
            s = states[t]
            a = path["actions"][t]
            s2 = states[t+1]
            r_a = env.r_a(s,a)
            r_s = env.r_s(s,a)
            f_s = env.f_s(s,a)
            f_a = env.f_a(s,a)
            pi_theta = policy.pi_theta(s)
            pi_s = policy.pi_s(s)

            V_s[t] = r_s + r_a.dot(pi_s) + gamma * V_s[t+1].dot(f_s + f_a.dot(pi_s))
            V_theta[t] = r_a.dot(pi_theta) + gamma * V_s[t+1].dot(f_a.dot(pi_theta)) + gamma * V_theta[t+1]
        return V_s,V_theta

    def train(self):
        self.start_worker()
        for itr in range(self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                path = rollout(
                    self.env,
                    self.policy,
                    self.max_path_length,
                    record_full_state=True,
                )
                logger.record_tabular("Iteration",itr)
                logger.record_tabular("Return",np.sum(path["rewards"]))
                self.optimize_policy(itr, path)
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr)
                params["algo"] = self
                params["path"] = path
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")
        self.shutdown_worker()


    def start_worker(self):
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        pass

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def get_itr_snapshot(self, itr):
        return self.itr_data

    def optimize_policy(self,itr,path):
        V_s, V_theta = self.compute_policy_gradient(
            self.env,
            self.policy,
            path,
            gamma=self.discount,
        )
        grad = V_theta[0]
        self.policy.set_theta(self.policy.get_theta() + self.lr * grad)
        self.itr_data = dict(
            V_s=V_s,
            V_theta=V_theta,
        )
