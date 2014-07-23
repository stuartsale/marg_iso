# Import stuff

import numpy as np
#import iso_lib as il
import sklearn as sk
import matplotlib as mpl
import matplotlib.pyplot as plt
import emcee




def emcee_prob(params, star):
    A=np.exp(params[4])
    dist=pow(10., params[3]/5.+1.)
            
    try:
        iso_obj=star.isochrones.query(params[0], params[1], params[2])
    except IndexError:
        return -1E9
        
    return -(np.power(star.r-(iso_obj.r0+params[3]+iso_obj.vr*A+iso_obj.ur*A*A) ,2)/(2*star.dr*star.dr)
            +np.power(star.i-(iso_obj.i0+params[3]+iso_obj.vi*A+iso_obj.ui*A*A) ,2)/(2*star.di*star.di)
            +np.power(star.ha-(iso_obj.ha0+params[3]+iso_obj.vha*A+iso_obj.uha*A*A) ,2)/(2*star.dha*star.dha) \
            +np.log(iso_obj.Jac) + 3*np.log(dist) - dist/2500. -2.3*np.log(iso_obj.Mi) + 2.3026*iso_obj.logage + param[4] #3*0.4605*(self.last_dist_mod+5) + self.last_logA - pow(10., self.last_dist_mod/5.+1.)/2500.  
            

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
        
        self.r_chain=np.zeros(chain_length)
        self.i_chain=np.zeros(chain_length)
        self.ha_chain=np.zeros(chain_length)
        
        
        
    # find likelihood of last param set
    
    def set_lastprob(self):
        A=np.exp(self.last_logA)
        self.last_prob=0.
        self.last_prob=-(np.power(self.r-(self.last_iso_obj.r0+self.last_dist_mod+self.last_iso_obj.vr*A+self.last_iso_obj.ur*A*A) ,2)/(2*self.dr*self.dr)
                        +np.power(self.i-(self.last_iso_obj.i0+self.last_dist_mod+self.last_iso_obj.vi*A+self.last_iso_obj.ui*A*A) ,2)/(2*self.di*self.di)
                        +np.power(self.ha-(self.last_iso_obj.ha0+self.last_dist_mod+self.last_iso_obj.vha*A+self.last_iso_obj.uha*A*A) ,2)/(2*self.dha*self.dha ))
            

    #  find likelihood of test param set

    def set_testprob(self):
        A=np.exp(self.test_logA)
        self.test_prob=0.
        self.test_prob=-(np.power(self.r-(self.test_iso_obj.r0+self.test_dist_mod+self.test_iso_obj.vr*A+self.test_iso_obj.ur*A*A) ,2)/(2*self.dr*self.dr)
                        +np.power(self.i-(self.test_iso_obj.i0+self.test_dist_mod+self.test_iso_obj.vi*A+self.test_iso_obj.ui*A*A) ,2)/(2*self.di*self.di)
                        +np.power(self.ha-(self.test_iso_obj.ha0+self.test_dist_mod+self.test_iso_obj.vha*A+self.test_iso_obj.uha*A*A) ,2)/(2*self.dha*self.dha) )


    # find prior prob of last param set
    
    def set_lastprior(self):
        dist=pow(10., self.last_dist_mod/5.+1.)
        self.last_prior=np.log(self.last_iso_obj.Jac) + 3*np.log(dist) - dist/2500. -2.3*np.log(self.last_iso_obj.Mi) + 2.3026*self.last_iso_obj.logage + self.last_logA #3*0.4605*(self.last_dist_mod+5) + self.last_logA - pow(10., self.last_dist_mod/5.+1.)/2500.
        
    # find prior prob of test param set
    
    def set_testprior(self):
        dist=pow(10., self.test_dist_mod/5.+1.)
        self.test_prior=np.log(self.test_iso_obj.Jac) + 3*np.log(dist) - dist/2500. -2.3*np.log(self.test_iso_obj.Mi) + 2.3026*self.test_iso_obj.logage + self.test_logA #3*0.4605*(self.test_dist_mod+5) + self.test_logA - pow(10., self.test_dist_mod/5.+1.)/2500.
    
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
                
                A=np.exp(self.last_logA)                
                self.r_chain[it/thin]=self.last_iso_obj.r0+self.last_dist_mod+self.last_iso_obj.vr*A+self.last_iso_obj.ur*A*A
                self.i_chain[it/thin]=self.last_iso_obj.i0+self.last_dist_mod+self.last_iso_obj.vi*A+self.last_iso_obj.ui*A*A
                self.ha_chain[it/thin]=self.last_iso_obj.ha0+self.last_dist_mod+self.last_iso_obj.vha*A+self.last_iso_obj.uha*A*A
                
                if it!=0 and it%10000==0:
                    print self.accept*1./(it+1)
                    
                    
    # ==============================================================
    # Functions to work with emcee sampler
    
    def emcee_init(self, N_walkers, chain_length):
    
        self.start_params=np.zeros([N_walkers,5])
    
        self.guess_set=[]
        guess_set.append([0.,3.663 ,4.57 ,0.,0.]);	#K4V
        guess_set.append([0.,3.672 ,4.56 ,0.,0.]);	#K3V
        guess_set.append([0.,3.686 ,4.55 ,0.,0.]);	#K2V
        guess_set.append([0.,3.695 ,4.55 ,0.,0.]);	#K1V
        guess_set.append([0.,3.703 ,4.57 ,0.,0.]);	#K0V
        guess_set.append([0.,3.720 ,4.55 ,0.,0.]);	#G8V
        guess_set.append([0.,3.740 ,4.49 ,0.,0.]);	#G5V
        guess_set.append([0.,3.763 ,4.40 ,0.,0.]);	#G2V
        guess_set.append([0.,3.774 ,4.39 ,0.,0.]);	#G0V
        guess_set.append([0.,3.789 ,4.35 ,0.,0.]);	#F8V
        guess_set.append([0.,3.813 ,4.34 ,0.,0.]);	#F5V
        guess_set.append([0.,3.845 ,4.26 ,0.,0.]);	#F2V
        guess_set.append([0.,3.863 ,4.28 ,0.,0.]);	#F0V
        guess_set.append([0.,3.903 ,4.26 ,0.,0.]);	#A7V
        guess_set.append([0.,3.924 ,4.22 ,0.,0.]);	#A5V
        guess_set.append([0.,3.949 ,4.20 ,0.,0.]);	#A3V
        guess_set.append([0.,3.961 ,4.16 ,0.,0.]);	#A2V
        
        guess_set.append([0.,3.763 ,3.20 ,0.,0.]);	#G2III
        guess_set.append([0.,3.695 ,2.95 ,0.,0.]);	#G8III
        guess_set.append([0.,3.663 ,2.78 ,0.,0.]);	#K1III
        guess_set.append([0.,3.602 ,1.93 ,0.,0.]);	#K5III
        guess_set.append([0.,3.591 ,1.63 ,0.,0.]);	#M0III
        
        for i in range(len(guess_set)):
            iso_obj=self.isochrones.query(guess_set[i][0], guess_set[i][1], guess_set[i][2])
            guess_set[i][4]=((self.r-self.i)-(iso_obj.r0-iso_obj.i0))/(iso_obj.vr-iso_obj.vi)
            guess_set[i][3]=self.r- iso_obj.vr*guess_set[i][4]+iso_obj.ur*guess_set[i][4]*guess_set[i][4]
            
        for it in range(N_walkers):
            self.start_params[i,:]=guess_set[int(np.random.uniform()*len(guess_set))]
            
        self.Teff_chain=np.zeros(chain_length)
        self.logg_chain=np.zeros(chain_length)
        self.feh_chain=np.zeros(chain_length)
        self.dist_mod_chain=np.zeros(chain_length)
        self.logA_chain=np.zeros(chain_length)

        self.prob_chain=np.zeros(chain_length)
        self.prior_chain=np.zeros(chain_length)
        self.Jac_chain=np.zeros(chain_length)
        
        self.r_chain=np.zeros(chain_length)
        self.i_chain=np.zeros(chain_length)
        self.ha_chain=np.zeros(chain_length)            
            
            
    def emcee_run(self, iterations=1000, thin=1, burn_in=100, N_walkers=30):
    
        self.emcee_init(N_walkers, (iterations-burn_in)/thin*N_walkers)
    
        sampler = emcee.EnsembleSampler(nwalkers, 5, emcee_prob)
        
        pos, last_prob, state = sampler.run_mcmc(self.start_params, burn_in)     # Burn-in
        sampler.reset()
        
        for i in range((iterations-burn_in)/thin):
            pos, last_prob, state=sampler.sample(pos, last_prob, state, iterations=thin, storechain=False)       # proper run
            
            self.feh_chain[i*N_walkers:(i+1)*N_walkers]=pos[:,0]
            self.Teff_chain[i*N_walkers:(i+1)*N_walkers]=pos[:,1]
            self.logg_chain[i*N_walkers:(i+1)*N_walkers]=pos[:,2]
            self.dist_mod_chain[i*N_walkers:(i+1)*N_walkers]=pos[:,3]
            self.logA_chain[i*N_walkers:(i+1)*N_walkers]=pos[:,4]
            
            self.prob_chain[i*N_walkers:(i+1)*N_walkers]=  last_prob  


                
    # ==============================================================
    # Auxilary functions
    
    # plot the MCMC sample on the ln(s) ln(A) plane
    
    def plot_MCMCsample(self):
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        
        ax1.scatter(self.dist_mod_chain, self.logA_chain, marker='.')
        plt.show()
        
    def chain_dump(self, filename):
        X=np.array( [np.arange(0,len(self.Teff_chain)), self.Teff_chain, self.logg_chain, self.feh_chain, self.dist_mod_chain, self.logA_chain, self.prob_chain, self.prior_chain, self.Jac_chain, self.r_chain, self.i_chain, self.ha_chain ]).T
        np.savetxt(filename, X, header="N\tTeff\tlogg\tfeh\tdist_mod\tlogA\tlike\tprior\tJac\tr\ti\tha\n" )
    
    # Fit Gaussians
    
#    def gauss_fit(self):
