
"""

Custom prior classes go here

all classess should have an __init__ function for adding self parameters, as well as a .sample() method

"""


class prior_uniform:
    """
    generate from uniform prior over interval [low,high] using the .sample() method
    """
    def __init__(self,low=0.,high=1.):
        self.low = low
        self.high = high
    
    def sample(self):
        return np.random.uniform()*(self.high-self.low) + self.low