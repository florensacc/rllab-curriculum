import numpy as np

class VI(object):
    def __init__(self, p, r, gamma):
        self.p = p
        self.r = r
        self.gamma = gamma

        ns, na, ns_next = p.shape
        assert ns == ns_next and r.shape == (ns,na)
        self.ns = ns
        self.na = na

    def v_iter(self, n_itr, soft, V=None):
        """
        V(s) = softmax_a(r(s,a) + gamma * E_{s'} V(s'))
        softmax_a = log (sum_a (exp(...)))
        """
        if V is None:
            V = np.zeros(self.ns)
        for i in range(n_itr):
            prev_V = np.expand_dims(
                np.expand_dims(V, axis=0),
                axis=0,
            ) # 1 x 1 x ns
            next_V = np.sum(self.p * prev_V, axis=2) # ns x na

            targets = self.r + self.gamma * next_V # ns x na
            if soft:
                targets_max = np.amax(targets, axis=1, keepdims=True) # ns
                V = np.log(np.sum(np.exp(targets - targets_max), axis=1)) + \
                    targets_max.ravel()
            else:
                V = np.amax(
                    targets,
                    axis=1,
                )
            Q = self.get_Q(V)
        return V

    def pi_iter(self, n_itr, V=None, beta=1):
        if V is None:
            V = np.zeros(self.ns)
        for i in range(n_itr):
            prev_V = np.expand_dims(
                np.expand_dims(V, axis=0),
                axis=0,
            )
            next_V = np.sum(self.p * prev_V, axis=2)

            targets = self.r + self.gamma * next_V
            Q = self.get_Q(V)
            pi = self.get_pi_from_Q(Q, beta)
            new_V = np.sum(pi * targets, axis=1)
            abs_error = np.mean((new_V - V) ** 2)
            TINY = 1e-15
            rel_error = abs_error / np.mean(V ** 2 + TINY)
            V = new_V
        print("relative error: ", rel_error)
        return V, Q, pi

    def get_pi_from_Q(self, Q, beta):
        Q = Q - np.amax(Q, axis=1, keepdims=True)
        un_probs = np.exp(Q * beta)
        pi = un_probs / np.sum(un_probs, axis=1, keepdims=True)
        return pi

    def get_Q(self, V):
        """
        Q(s,a) = r(s,a) + gamma * E_{s'} V(s')
        """
        assert V.shape == (self.ns,)
        V_expanded = np.expand_dims(np.expand_dims(V,axis=0),axis=0)
        Q = self.r + self.gamma * np.sum(self.p * V_expanded, axis=2)
        return Q

    def get_soft_policy(self, V, Q):
        assert V.shape == (self.ns,) and Q.shape == (self.ns, self.na)
        un_pi = np.exp(Q - np.expand_dims(V, axis=1))
        pi = un_pi / np.sum(un_pi, axis=1, keepdims=True)
        return pi

    def get_policy(self, Q):
        opt_actions = np.argmax(Q, axis=1)
        pi = np.zeros_like(Q)
        ns, na = Q.shape
        pi[np.arange(ns), opt_actions] = 1
        return pi
