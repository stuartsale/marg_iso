# Import stuff

import numpy as np
#import iso_lib as il
import sklearn as sk


# Class to contain star's data, chain, etc

class star_posterior:

    # init function
    
    def __init__(self, r_in, i_in, ha_in, dr_in, di_in, dha_in, isochrones):
    
        self.r=r_in
        self.i=i_in
        self.ha=ha_in
        
        self.dr=dr_in
        self.di=di_in
        self.dha=dha_in
        
        self.isochrones=isochrones
        
    # initialise MCMC_chain
    
    def init_chain(self, chain_length):
        self.last_Teff=3.8
        self.last_logg=4.5
        self.last_feh=0.
        self.last_dist_mod=10.
        self.last_logA=1.
        
        self.last_iso_obj=isochrones.query(self.last_feh, self.last_Teff, self.last_logg)
        
        self.set_lastprob()
        
        self.Teff_chain=np.zeros(chain_length)
        self.logg_chain=np.zeros(chain_length)
        self.feh_chain=np.zeros(chain_length)
        self.dist_mod_chain=np.zeros(chain_length)
        self.logA_chain=np.zeros(chain_length)
        
        
        
    # find probability of last param set
    
    def set_lastprob(self):
        self.last_prob=0.    
#        self.last_prob=(np.power(self.r- ,2)
#                        +np.power(self.i- ,2)
#                        +np.power(self.ha- ,2) )
            

    #  find probability of test param set

    def set_testprob(self):
        self.test_prob=0.
#        self.last_prob=(np.power(self.r- ,2)
#                        +np.power(self.i- ,2)
#                        +np.power(self.ha- ,2) )
    
    # MCMC sampler
    
    def MCMC_Metropolis(self, iterations=10000, thin=1, Teff_prop, logg_prop, feh_prop, dist_mod_prop, logA_prop):
    
        self.init_chain(int(iterations/thin))
        
        for it in range(iterations):
            # Set test params
            
            test_Teff=3.8
            test_logg=4.5
            test_feh=0.
            test_dist_mod=10.
            test_logA=1.
            
            test_iso_obj=isochrones.query(self.test_feh, self.test_Teff, self.test_logg)    
            
            # get probs
            
            set_testprob()
            
            # accept/reject
            
            threshold=np.log(np.random.uniform())
            
            if (self.test_prob-self.last_prob)>=threshold:
                self.last_Teff=test_Teff
                self.last_logg=test_logg
                self.last_feh=test_feh
                self.last_dist_mod=test_dist_mod
                self.last_logA=test_logA
                        
                self.last_iso_obj=test_iso_obj
            
            # store in chain 
            
            if it%thin==0:   
                self.Teff_chain[it/thin]=self.last_Teff
                self.logg_chain[it/thin]=self.last_logg
                self.feh_chain[it/thin]=self.last_feh
                self.dist_mod_chain[it/thin]=self.last_dist_mod
                self.logA_chain[it/thin]=self.last_logA
                
                print it
    
    # Fit Gaussians
    
#    def gauss_fit(self):
