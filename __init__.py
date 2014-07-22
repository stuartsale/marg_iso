# Import stuff

import numpy as np
#import iso_lib as il
import sklearn as sk
import matplotlib as mpl
import matplotlib.pyplot as plt


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
        self.last_logg=4.3
        self.last_feh=0.
        self.last_dist_mod=10.
        self.last_logA=0.
        
        self.last_iso_obj=self.isochrones.query(self.last_feh, self.last_Teff, self.last_logg)
        
        self.set_lastprob()
        self.set_lastprior()
        
        self.Teff_chain=np.zeros(chain_length)
        self.logg_chain=np.zeros(chain_length)
        self.feh_chain=np.zeros(chain_length)
        self.dist_mod_chain=np.zeros(chain_length)
        self.logA_chain=np.zeros(chain_length)

        self.prob_chain=np.zeros(chain_length)
        self.prior_chain=np.zeros(chain_length)
        self.Jac_chain=np.zeros(chain_length)
        
        
        
    # find likelihood of last param set
    
    def set_lastprob(self):
        A=np.exp(self.last_logA)
        self.last_prob=0.
        self.last_prob=-(np.power(self.r-(self.last_iso_obj.r0+self.last_dist_mod+self.last_iso_obj.vr*A+self.last_iso_obj.ur*A*A) ,2)/self.dr
                        +np.power(self.i-(self.last_iso_obj.i0+self.last_dist_mod+self.last_iso_obj.vi*A+self.last_iso_obj.ui*A*A) ,2)/self.di
                        +np.power(self.ha-(self.last_iso_obj.ha0+self.last_dist_mod+self.last_iso_obj.vha*A+self.last_iso_obj.uha*A*A) ,2)/self.dha )
            

    #  find likelihood of test param set

    def set_testprob(self):
        A=np.exp(self.test_logA)
        self.test_prob=0.
        self.test_prob=-(np.power(self.r-(self.test_iso_obj.r0+self.test_dist_mod+self.test_iso_obj.vr*A+self.test_iso_obj.ur*A*A) ,2)/self.dr
                        +np.power(self.i-(self.test_iso_obj.i0+self.test_dist_mod+self.test_iso_obj.vi*A+self.test_iso_obj.ui*A*A) ,2)/self.di
                        +np.power(self.ha-(self.test_iso_obj.ha0+self.test_dist_mod+self.test_iso_obj.vha*A+self.test_iso_obj.uha*A*A) ,2)/self.dha )

    # find prior prob of last param set
    
    def set_lastprior(self):
        self.last_prior=np.log(self.last_iso_obj.Jac)
        
    # find prior prob of test param set
    
    def set_testprior(self):
        self.test_prior=np.log(self.test_iso_obj.Jac)
    
    # MCMC sampler
    
    def MCMC_Metropolis(self, Teff_prop, logg_prop, feh_prop, dist_mod_prop, logA_prop, iterations=10000, thin=1):
    
        self.init_chain(int(iterations/thin))
        
        self.accept=0
        
        for it in range(iterations):
            # Set test params
            
            self.test_Teff=self.last_Teff + np.random.normal()*Teff_prop
            self.test_logg=self.last_logg + np.random.normal()*logg_prop
            self.test_feh=self.last_feh + np.random.normal()*feh_prop
            self.test_dist_mod=self.last_dist_mod + np.random.normal()*dist_mod_prop
            self.test_logA=self.last_logA + np.random.normal()*logA_prop
            
            try:
                self.test_iso_obj=self.isochrones.query(self.test_feh, self.test_Teff, self.test_logg)    
            except IndexError:
                continue
            
            # get probs
            
            self.set_testprob()
            self.set_testprior()
            
            # accept/reject
            
            threshold=np.log(np.random.uniform())
            
            if (self.test_prob-self.last_prob+self.test_prior-self.last_prior)>=threshold:
                self.last_prob=self.test_prob
                self.last_prior=self.test_prior

                self.last_Teff=self.test_Teff
                self.last_logg=self.test_logg
                self.last_feh=self.test_feh
                self.last_dist_mod=self.test_dist_mod
                self.last_logA=self.test_logA
                        
                self.last_iso_obj=self.test_iso_obj
                
                self.accept+=1
                
#            else:
#                print self.test_prob, self.last_prob
            
            # store in chain 
            
            if it%thin==0:   
                self.Teff_chain[it/thin]=self.last_Teff
                self.logg_chain[it/thin]=self.last_logg
                self.feh_chain[it/thin]=self.last_feh
                self.dist_mod_chain[it/thin]=self.last_dist_mod
                self.logA_chain[it/thin]=self.last_logA 

                self.prob_chain[it/thin]=self.last_prob 
                self.prior_chain[it/thin]=self.last_prior 
                self.Jac_chain[it/thin]=self.last_iso_obj.Jac 
               

                
                if it!=0:
                    print self.accept*1./(it+1)
                

    # plot the MCMC sample on the ln(s) ln(A) plane
    
    def plot_MCMCsample(self):
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        
        ax1.scatter(self.dist_mod_chain, self.logA_chain, marker='.')
        plt.show()
        
    def chain_dump(self, filename):
        X=np.array( [np.arange(0,len(self.Teff_chain)), self.Teff_chain, self.logg_chain, self.feh_chain, self.dist_mod_chain, self.logA_chain, self.prob_chain, self.prior_chain, self.Jac_chain ]).T
        np.savetxt(filename, X, header="N\tTeff\tlogg\tfeh\tdist_mod\tlogA\tlike\tprior\tJac\n" )
    
    # Fit Gaussians
    
#    def gauss_fit(self):
