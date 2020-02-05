from ABC_code import *

class distance:
    
    """
    distance class
    args: 
        -reference graph 'G0' (igraph object)
        -proposal graph 'G_prop' generated from model (igraph object)
        -function 'measure' taking two igraph objects and returning a scalar 
        -'**measure_kwargs' are optional keyword args that may apply to 'measure'
    """
    
    def __init__(self,G0,measure,**measure_kwargs):
        
        """
        distance class attributes
        """
        
        self.measure = measure
        self.G0 = G0
        self.measure_kwargs = measure_kwargs
        
    def evaluate_unnormalized(self,G_prop):
        
        """
        evaluate unnormalized distance between G0 and igraph object G_prop
        """
        self.G_prop = G_prop
        
        return self.measure(self.G0,self.G_prop,**self.measure_kwargs)
    
    def evaluate_normalized(self,G_prop,num_nulls,null_generator,**null_kwargs): 
        
        """
        evaluate normalized distance between G0 and igraph object G_prop
        args:
            -'num_nulls' is number of null graphs to generate to compute z-score
            -'null_generator' is function that takes in a reference graph G0 and outputs an igraph object that is randomized
                    based on some property of G0 and some model/parameters included in **null_kwargs
        """
        
        null_distance_vals = [self.measure(self.G0,null_generator(self.G0,**null_kwargs),**self.measure_kwargs)\
                              for _ in range(num_nulls)]
        null_mean,null_std = np.mean(null_distance_vals),np.std(null_distance_vals)
        
        return (self.evaluate_unnormalized(G_prop) - null_mean)/null_std
    

    
"""

Custom distance functions go here

args:
    -G0: reference graph in igraph format
    -G_prop: proposed graph in igraph format
    -any other args should be named and set to some default value (i.e. beta=1 in the first example)

"""    

def distance_KS_degree(G0,G_prop,beta=1.):
    """
    Kolmogorov-Smirnov distance between distributions of degrees to the 'beta' power
    """
    return ss.ks_2samp(np.array(G0.vs.degree())**beta,np.array(G_prop.vs.degree())**beta).statistic
