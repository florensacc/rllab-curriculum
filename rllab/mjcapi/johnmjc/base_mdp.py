class MDP(object):

    def call(self,input_arrs):
        """
        Call the mdp
        """
        raise NotImplementedError

    def initialize_mdp_arrays(self):
        raise NotImplementedError

    def input_info(self):
        """
        mapping from name -> (size, dtype)
        """
        raise NotImplementedError

    def output_info(self):
        raise NotImplementedError

    def plot(self, input_arrs, output_arrs):
        raise NotImplementedError

    def cost_names(self):
        raise NotImplementedError

    ################################

    def unscaled_cost(self, cost_dict):
        return {}

    def input_size(self,name):
        return self.input_info()[name][0]        

    def input_dtype(self,name):
        return self.input_info()[name][1]

    def output_size(self,name):
        return self.output_info()[name][0]        

    def output_dtype(self,name):
        return self.output_info()[name][1]

    def num_costs(self):
        return len(self.cost_names())

    def validate(self):
        print "---------------------------"
        print "validating mdp %s"%self.__class__.__name__
        init_arrs = self.initialize_mdp_arrays()
        print "init_keys",init_arrs.keys()
        assert "c" not in init_arrs
        assert "o" in init_arrs
        input_info = self.input_info()
        print "input_info",input_info
        assert "u" in input_info
        output_info = self.output_info()
        print "output_info",output_info        
        assert "o" in output_info
        assert "c" in output_info
        arrs= self.initialize_mdp_arrays()
        import numpy as np
        from control4.config import floatX
        arrs["u"] = np.zeros((1,self.input_size("u")),self.input_dtype("u"))
        newarrs = self.call(arrs)
        assert newarrs["c"].dtype == floatX and newarrs["c"].shape[0] == 1
        print "---------------------------"
