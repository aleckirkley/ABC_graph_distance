from ABC_code import *

"""

Custom Hamiltonians go here

args:
    -dist: distance from proposed graph to reference graph
    -eps: characteristic distance scale over which we accept things
    -any other args should be named and set to some default value (i.e. barrier = 1e100 in the first example)

"""


def h_hard_cutoff(dist,eps,barrier=1e100):
    """
    hamiltonian with infinite barrier at threshold eps
    """
    if dist < eps:
        return -barrier
    else:
        return barrier