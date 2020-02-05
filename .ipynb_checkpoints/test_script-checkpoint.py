from ABC_code import *
import matplotlib.pyplot as plt
from time import sleep

"""
Generate initial reference graph
"""
N = 1000
M = 10
true_alpha = 1.1
G0 = generator_PA(N, M, alpha = true_alpha) #true alpha is 1.1 here


"""
Create 'distance' object using reference graph and distance function of interest
"""
distance_function = distance_KS_degree
distance_obj = distance(G0, distance_function, beta = 1.)


"""
Define graph generator, free parameters to infer, and priors on these free parameters
"""
generator_function = generator_PA #we use generalized PA in this example
free_parameters_dict = {'alpha': -1.} #set some initial value for the free parameters. it shouldn't matter
priors_dict = {'alpha': prior_uniform(low = 0.5, high = 1.5)} #we do not have a great guess at alpha here (0.5-1.5)


"""
Create 'model' object using a graph generator, a prior, and the free parameters to infer
    Remember to use keywords for all fixed parameters (here, N and M)
    Make sure that all parameters for 'generator' are specified, either as free (here, alpha) or fixed (here, N and M) 
    Other arguments for the priors should be contained in those object definitions (as shown above for 'low' and 'high')
"""
model_obj = model(generator_function, free_parameters_dict, priors_dict, N = N, M = M)


"""
Create 'abc' object using:
    -desired acceptance rate (only will be used if you run the 'get_eps_from_acc' method)
    -model object fully specified with a generator, free parameters, and priors
    -distance object fully specified with reference graph and distance function
    -epsilon (distance threshold). can be set to 'None' if you want to train for this using 'acc'
    -hamiltonian function that takes args (distance value, epsilon value, other args)
        -the other args for the hamiltonian should be specified as separate keyword args (i.e. barrier=1e200 here)

"""
hamiltonian = h_hard_cutoff
acc = 0.05 #we will use a 5% acceptance rate in this example
abc_obj = ABC(acc = acc, model_object = model_obj, distance_object = distance_obj, eps = 'None', hamiltonian = hamiltonian, \
             barrier = 1e200)

print('Created abc object...')

"""
Find the epsilon value associated with a desired acceptance rate
Sets abc_obj.eps equal to the result
args:
    -tol: how close to the desired 'acc' value we want to be ('acc' is specified in the abc object definition)
    -max_runs: maximum number of iterations of bisection to estimate epsilon if we cannot get to within 'tol' of 'acc'
    -eps_lower/upper_init: initial guesses that we are certain will be below/above the correct epsilon value
        -just running a few test simulations we can make these bounds 
        -very loose bounds will work, but cause it to take a long time to converge, so tighter bounds are best
    -window_size: how many runs used to estimate the acceptance rate for a given guess of epsilon
    -num_sims: number of reruns for the estimate of epsilon. more runs reduces the effect of poor convergence
"""
abc_obj.get_eps_from_acc(max_runs=20, tol=.0001, eps_lower_init=0., eps_upper_init=1., window_size=100, num_sims=2)

print('Retrieved an epsilon value of',abc_obj.eps,' for a desired acceptance rate of',acc*100,'%...')

"""
Run ABC simulations 
n_runs is number of proposals to be made
All parameters are stored in abc_obj, and can be accessed by 
    abc_obj.acc - desired accuracy, in case we forgot
    abc_obj.model - model object
    abc_obj.distance - distance object
    abc_obj.G0 - reference graph
    abc_obj.eps - currently stored epsilon value
                    after training with 'get_eps_from_acc' this is the value that will be stored
                    if no training happened, it will default to whatever is input as 'eps' in the function definition
    abc_obj.hamiltonian - hamiltonian function used
    abc_obj.hamiltonian_kwargs - keywords for the hamiltonian (beyond dist and eps)
    abc_obj.posterior - dict of {'parameter name': [accepted parameter values]}
    abc_obj.acceptance_rate - acceptance rate after most recent run of .run_sims() method
                                hopefully will be close to 'acc' if epsilon is trained correctly
"""
abc_obj.run_sims(n_runs = 2000)

print('Sampled posterior with epsilon value of ',abc_obj.eps,' for an actual acceptance rate of',abc_obj.acceptance_rate*100,'%...')

"""
Save posterior results to csv file
**save_kwargs in function definition are keyword arguments for the pandas.DataFrame.to_csv() function (here, index=False)
"""
filepath = 'test_posterior.csv'
abc_obj.save_to_file(filepath, index = False)
sleep(2)
print('and saved posterior results successfully to csv file!')

"""
Plot posterior results for this example
"""
sleep(2)
plt.hist(abc_obj.posterior['alpha'],bins=20,density=True)
plt.xlabel(r'$\alpha$')
plt.ylabel('Probability Density')
plt.show()