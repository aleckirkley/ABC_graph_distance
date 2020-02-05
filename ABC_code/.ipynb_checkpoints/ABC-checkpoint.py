from ABC_code import *

class ABC:
    """
    ABC simulator class
    args:
        -'acc': desired acceptance rate
        -'model_object': model object
        -'distance_object': distance object
        -'G0': reference graph to fit for posterior
        -'eps': numerical cutoff value (if applicable)
            -uses acceptance rate if eps == None (default)
        -'hamiltonian': hamiltonian function
            -h(d,eps) is a function of distance d and cutoff scale eps
            -uses hard thresholding hamiltonian by default
    """
    
    def __init__(self,acc,model_object,distance_object,eps='None',hamiltonian='None',**hamiltonian_kwargs):
        self.acc = acc
        self.model = model_object
        self.distance = distance_object
        self.G0 = self.distance.G0
        self.eps = eps
        self.hamiltonian = hamiltonian
        self.hamiltonian_kwargs = hamiltonian_kwargs
        
        self.posterior = dict.fromkeys(self.model.free_param_names)
        self.acceptance_rate = 'None'
        
    def get_eps_from_acc(self,max_runs='None',tol='None',eps_lower_init='None',eps_upper_init='None',\
                         window_size='None',num_sims='None'):
        """
        finds epsilon value that gives approximately the acceptance rate 'acc'
            for the given distance object
        stores this as self.eps
        takes moving average of acceptance rate and tries to get within 'tol' of 'acc'
        updates guess for epsilon by repeatedly bisecting an interval
        args:
            -'eps_init': initial guess for epsilon
            -'max_runs': number of runs algorithm allowed to run for if acc_current
                    never within 'tol' of 'acc'
            -'tol': tolerance of error for accepting acc_current
            -'eps_lower/upper_init': lower/upper cutoff for epsilon bisection
        """
        
        
        eps_currents = []
        for _ in range(num_sims):
            acc_current,run_count = 100,0
            eps_upper = eps_upper_init
            eps_lower = eps_lower_init
            proposal_results = [False]*window_size
            while (run_count < max_runs):

                eps_current = np.random.rand()*(eps_upper-eps_lower) + eps_lower

                for w in range(window_size):

                    G_prop = self.model.generate_graph()
                    params_prop = self.model.free_parameters

                    dist = self.distance.evaluate_unnormalized(G_prop)

                    if np.log(np.random.rand()) < -self.hamiltonian(dist,eps_current,**self.hamiltonian_kwargs):
                        proposal_results[w] = True
                    else:
                        proposal_results[w] = False

                acc_current = Counter(proposal_results)[True]/window_size

                if abs(acc_current-self.acc) > tol:
                    if acc_current < self.acc:
                        eps_lower = eps_current 
                    else:
                        eps_upper = eps_current
                    eps_current = (eps_upper + eps_lower)/2

                else:
                    break

                run_count += 1
            eps_currents.append(eps_current)
            
        self.eps = np.mean(eps_currents) 
        
    def run_sims(self,n_runs='None'):
        """
        runs ABC with soft thresholding (hard thresholding imposed by the default hamiltonian h_hard_cutoff)
        n_runs is desired number of runs (proposals) for the simulation
        sets self.accepted_params and self.acceptance_rate
        """
        
        accepted_params = []
        proposal_results = []
        run_count = 0
        while (run_count < n_runs):
                
            G_prop = self.model.generate_graph()
            params_prop = copy.deepcopy(self.model.free_parameters)

            dist = self.distance.evaluate_unnormalized(G_prop)

            if np.log(np.random.rand()) < -self.hamiltonian(dist,self.eps,**self.hamiltonian_kwargs):
                proposal_results.append(True)
                accepted_params.append(params_prop)
            else:
                proposal_results.append(False) 
                    
            run_count += 1 
        
        params = self.model.free_param_names
        posterior_dict = dict.fromkeys(params)
        for p in params:
            posterior_dict[p] = [res[p] for res in accepted_params]
        self.posterior = posterior_dict
        self.acceptance_rate = Counter(proposal_results)[True]/n_runs
    
    def save_to_file(self,filepath,**save_kwargs):
        """
        write to csv with column names as parameter names and column values as the 
            sampled values of that parameter
        """
        try:
            df = pd.DataFrame.from_dict(self.posterior)
            df.to_csv(filepath,**save_kwargs)
        except:
            print('No posterior data to write!')
        