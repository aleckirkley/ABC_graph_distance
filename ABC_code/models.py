from ABC_code import *

class model:
    
    """
    model class
    args:
        -graph generator 'generator' that takes in some named arguments and returns an igraph object
        -'free_parameters' is dict of named parameters to infer posteriors for
        -'priors' is dict of prior classes using same keys as 'free_parameters'
        -'**fixed parameters' is all other named parameters of 'generator' that we fix 
        fixed + free parameters need to fully specify 'generator'
    """
    
    def __init__(self,generator,free_parameters,priors,**fixed_parameters):
        self.generator = generator
        self.fixed_parameters = fixed_parameters
        self.priors = priors
        self.free_parameters = free_parameters
        
        self.free_param_names = list(self.free_parameters.keys())
    
    def prior_sample(self,param):
        """
        sample from prior associated with parameter 'param'
        """
        sample_class = self.priors[param]
        return sample_class.sample()
    
    def generate_graph(self):
        """
        generate graph using draw from prior, which is stored in self.free_parameters 
        """
        for param in self.free_param_names:
            self.free_parameters[param] = self.prior_sample(param)
        return self.generator(**self.free_parameters,**self.fixed_parameters)
    
    

"""

Custom models go here

args:
    -any args should be named and set to some default value (i.e. N=100, M=3, alpha=1 in the first example)


""" 

def generator_PA(N=100,M=3,alpha=1.):
    """
    generalized preferential attachment with exponent alpha, number of nodes and edges N,M
    """
    return ig.Graph().Barabasi(n=N, m=M, power=alpha)