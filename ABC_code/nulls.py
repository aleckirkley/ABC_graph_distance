
"""

Custom null model generators go here

args:
    -G0: reference graph to generate null ensemble draw from
    -all other args should be named and set to some default value (i.e. fix_edge_count=True in the first example)

"""

def null_ER(G0,fix_edge_count=True):
    """
    returns random graph with same node and edge counts as G0
    """
    N,M = G0.ecount(),G0.vcount()
    
    if fix_edge_count == True:
        return ig.Graph().Erdos_Renyi(n=N,m=M)
    
    else:
        return ig.Graph().Erdos_Renyi(n=N,p=2*M/(N*(N-1)))
        