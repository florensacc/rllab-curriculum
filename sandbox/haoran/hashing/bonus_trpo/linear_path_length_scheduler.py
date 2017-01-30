class LinearPathLengthScheduler(object):
    def __init__(self, init_len, final_len):
        self.init_len = init_len
        self.final_len = final_len

    def set_algo(self,algo):
        self.algo = algo

    def update(self, itr):
        cur_len = min(
            self.final_len,
            self.init_len + (self.final_len - self.init_len) * itr / self.algo.n_itr
        )
        self.algo.max_path_length = cur_len
        print("Updated path length to %d"%(cur_len))
