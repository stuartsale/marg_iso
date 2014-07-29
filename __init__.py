# Import stuff

import numpy as np
#import iso_lib as il
import sklearn as sk
import sklearn.cluster as sk_c
import sklearn.mixture as sk_m
import matplotlib as mpl
import matplotlib.pyplot as plt
import emcee

from scipy import linalg



def emcee_prob(params, star):
    A=np.exp(params[4])
    dist=pow(10., params[3]/5.+1.)
            
    try:
        iso_obj=star.isochrones.query(params[0], params[1], params[2])
    except IndexError:
        return -np.inf
        
    if iso_obj.Jac==0:
        return -np.inf
        
    else:
        return -(np.power(star.r-(iso_obj.r0+params[3]+iso_obj.vr*A+iso_obj.ur*A*A) ,2)/(2*star.dr*star.dr)
                +np.power(star.i-(iso_obj.i0+params[3]+iso_obj.vi*A+iso_obj.ui*A*A) ,2)/(2*star.di*star.di)
                +np.power(star.ha-(iso_obj.ha0+params[3]+iso_obj.vha*A+iso_obj.uha*A*A) ,2)/(2*star.dha*star.dha) ) \
                +np.log(iso_obj.Jac) + 3*np.log(dist) - dist/2500. -2.3*np.log(iso_obj.Mi) + 2.3026*iso_obj.logage + params[4] -np.power(params[0]+((8000+dist)-8000.)*0.00007,2)/(2*0.0625)
                #3*0.4605*(self.last_dist_mod+5) + self.last_logA - pow(10., self.last_dist_mod/5.+1.)/2500.  
                

# Class to contain star's data, chain, etc            

class star_posterior:

    # init function
    
    def __init__(self, r_in, i_in, ha_in, dr_in, di_in, dha_in, isochrones):
    
        self.colors = np.array([x for x in 'bgrcmybgrcmybgrcmybgrcmy'])
        self.colors = np.hstack([self.colors] * 20)
    
        self.r=r_in
        self.i=i_in
        self.ha=ha_in
        
        self.dr=dr_in
        self.di=di_in
        self.dha=dha_in
        
        self.isochrones=isochrones
        self.MCMC_run=False 
        self.best_gmm=None       
        
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
        
        self.itnum_chain=np.zeros(chain_length)                    
        
        
        
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
                
                self.itnum_chain[it/thin]=it
                
                if it!=0 and it%10000==0:
                    print self.accept*1./(it+1)
                    
                    
    # ==============================================================
    # Functions to work with emcee sampler
    
    def emcee_init(self, N_walkers, chain_length):
    
        self.start_params=np.zeros([N_walkers,5])
    
        guess_set=[]
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
        
#        guess_set.append([0.,3.763 ,3.20 ,0.,0.]);	#G2III
#        guess_set.append([0.,3.700 ,2.75 ,0.,0.]);	#G8III
#        guess_set.append([0.,3.663 ,2.52 ,0.,0.]);	#K1III
#        guess_set.append([0.,3.602 ,1.25 ,0.,0.]);	#K5III
#        guess_set.append([0.,3.591 ,1.10 ,0.,0.]);	#M0III
        
        for i in range(len(guess_set)):
            iso_obj=self.isochrones.query(guess_set[i][0], guess_set[i][1], guess_set[i][2])
            guess_set[i][4]=((self.r-self.i)-(iso_obj.r0-iso_obj.i0))/(iso_obj.vr-iso_obj.vi)
            guess_set[i][3]=self.r- iso_obj.vr*guess_set[i][4]+iso_obj.ur*guess_set[i][4]*guess_set[i][4]
    
        metal_min=sorted(self.isochrones.metal_dict.keys())[0]
        metal_max=sorted(self.isochrones.metal_dict.keys())[-1]
            
        for it in range(N_walkers):
            self.start_params[it,:]=guess_set[int(np.random.uniform()*len(guess_set))]
            self.start_params[it,0]=metal_min+(metal_max-metal_min)*np.random.uniform()
            
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
        
        self.itnum_chain=np.zeros(chain_length)            
            
            
    def emcee_run(self, iterations=10000, thin=10, burn_in=2000, N_walkers=50, prune=True, cluster_plot=False):
    
        self.emcee_init(N_walkers, (iterations-burn_in)/thin*N_walkers)
    
        sampler = emcee.EnsembleSampler(N_walkers, 5, emcee_prob, args=[self])
        
        pos, last_prob, state = sampler.run_mcmc(self.start_params, burn_in)     # Burn-in
        sampler.reset()
 
        if prune:        
            dbscan = sk_c.DBSCAN(eps=0.05)
            pos, last_prob, state = sampler.run_mcmc(pos, 10, rstate0=state, lnprob0=last_prob)     # pruning set
            dbscan.fit(sampler.flatchain[:,1:2])
            labels=dbscan.labels_.astype(np.int)
            
            if cluster_plot:

                fig=plt.figure()
                ax1=fig.add_subplot(111)
                
                print "num points = ",sampler.flatchain[:,1].size
                
                ax1.scatter(sampler.flatchain[:,1], sampler.flatchain[:,2], color=self.colors[labels].tolist(),)
                plt.show()
            
            mean_ln_prob=np.mean(sampler.flatlnprobability)
            cl_list=[]
            weights_list=[]
            weights_sum=0
            for cl_it in range(np.max(labels)+1):
                cl_list.append(posterior_cluster(sampler.flatchain[labels==cl_it,:], sampler.flatlnprobability[labels==cl_it]-mean_ln_prob))
                weights_sum+= cl_list[-1].weight
                weights_list.append(cl_list[-1].weight)
            print weights_sum
            
            for i in range(N_walkers):
                cluster=np.random.choice(np.max(labels)+1, p=weights_list/np.sum(weights_list))
                index=int( np.random.uniform()*len(cl_list[cluster]) )
                pos[i,:]=cl_list[cluster].data[index,:]
            
            sampler.reset()

        
        for i, (pos, prob, rstate) in enumerate(sampler.sample(pos, iterations=(iterations-burn_in), storechain=False)):      # proper run
        
            if i%thin==0:
            
                self.feh_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,0]
                self.Teff_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,1]
                self.logg_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,2]
                self.dist_mod_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,3]
                self.logA_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,4]
                
                self.prob_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]= prob
                
                self.itnum_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=  i
                
                for n in range(N_walkers):
                    try:
                        iso_obj=self.isochrones.query(pos[n,0], pos[n,1], pos[n,2])
                        A=np.exp(pos[n,4])
                        self.r_chain[i/thin*N_walkers+n]=iso_obj.r0#+pos[n,3]+iso_obj.vr*A+iso_obj.ur*A*A
                        self.i_chain[i/thin*N_walkers+n]=iso_obj.i0#+pos[n,3]+iso_obj.vi*A+iso_obj.ui*A*A
                        self.ha_chain[i/thin*N_walkers+n]=iso_obj.ha0#+pos[n,3]+iso_obj.vha*A+iso_obj.uha*A*A
                        
                        
                    except IndexError:
                        print -1E9  
                        
        self.MCMC_run=True
                        
    # ==============================================================                        
    # Fit Gaussians
    
    def gmm_fit(self):
    
        if self.MCMC_run:
            fit_points=np.array([self.dist_mod_chain, self.logA_chain]).T
            print fit_points.shape
            best_bic=+np.infty
            for n_components in range(1,11):
                gmm = sk_m.GMM(n_components=n_components, covariance_type='full',min_covar=0.05)
                gmm.fit(fit_points)
                if gmm.bic(fit_points)<best_bic:
                    best_bic=gmm.bic(fit_points)
                    self.best_gmm=gmm
                    print n_components, best_bic, np.sort(gmm.weights_), "*"
                else:
                    print n_components, gmm.bic(fit_points), np.sort(gmm.weights_)
                    
    def gmm_sample(self, filename=None, num_samples=None):
        if self.best_gmm:
            if num_samples==None:
                num_samples=self.prob_chain.size
                
            components=np.random.choice(self.best_gmm.weights_.size, p=self.best_gmm.weights_, size=num_samples)
            
            covar_roots=[]
            for covar in self.best_gmm._get_covars():
                covar_roots.append(np.linalg.cholesky(covar))
            
            self.gmm_sample=self.best_gmm.means_[components]  #+ np.dot(self.best_gmm._get_covars()[0], np.random.normal(size=(2, num_samples)) ).T
            for it in range(components.size):
                self.gmm_sample[it,:]+=np.dot(covar_roots[components[it]], np.random.normal(size=(2, 1) )).flatten()
            
            if filename:
                np.savetxt(filename, self.gmm_sample)
            
                        
                
    # ==============================================================                        
    # Auxilary functions
    
    # plot the MCMC sample on the ln(s) ln(A) plane
    
    def plot_MCMCsample(self):
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        
        ax1.scatter(self.dist_mod_chain, self.logA_chain, marker='.')
        plt.show()

    # dump chain to text file
        
    def chain_dump(self, filename):
        X=np.array( [self.itnum_chain, self.Teff_chain, self.logg_chain, self.feh_chain, self.dist_mod_chain, self.logA_chain, self.prob_chain, self.prior_chain, self.Jac_chain, self.r_chain, self.i_chain, self.ha_chain ]).T
        np.savetxt(filename, X, header="N\tTeff\tlogg\tfeh\tdist_mod\tlogA\tlike\tprior\tJac\tr\ti\tha\n" )
    

    # plot MCMC sample overlaid with gaussian fit in dist_mod x log(A) space
    
    def plot_MCMCsample_gaussians(self):
        fit_points=np.array([self.dist_mod_chain, self.logA_chain]).T
        Y_=self.best_gmm.predict(fit_points)
        

    
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        
        for it in range(self.best_gmm.weights_.size):
            ax1.scatter(fit_points[Y_==it,0], fit_points[Y_==it,1], marker='.', color=self.colors[it])

            # Plot an ellipse to show the Gaussian component            
            
            v, w = linalg.eigh(self.best_gmm._get_covars()[it])
            
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180 * angle / np.pi  # convert to degrees
            v *= 4
            ell = mpl.patches.Ellipse(self.best_gmm.means_[it], v[0], v[1], 180 + angle, color=self.colors[it])
            ell.set_clip_box(ax1.bbox)
            ell.set_alpha(.5)
            ax1.add_artist(ell)
        plt.show()
        
    def compare_MCMC_hist(self):
        fig=plt.figure()
        ax1=fig.add_subplot(111) 
        
        bins=np.arange(8.,17.5, 0.25)
        
        ax1.hist(self.dist_mod_chain, bins, histtype='step', ec='k')
        
        x=np.arange(np.min(self.dist_mod_chain), np.max(self.dist_mod_chain), 0.1)
        y=np.zeros(x.size)
        
        for it in range(self.best_gmm.weights_.size):
             y+=1/np.sqrt(2*np.pi*self.best_gmm._get_covars()[it][0,0]) * np.exp(-np.power(x-self.best_gmm.means_[it][0],2)/(2*self.best_gmm._get_covars()[it][0,0]) ) * self.best_gmm.weights_[it]
             y_it=1/np.sqrt(2*np.pi*self.best_gmm._get_covars()[it][0,0]) * np.exp(-np.power(x-self.best_gmm.means_[it][0],2)/(2*self.best_gmm._get_covars()[it][0,0]) ) * self.dist_mod_chain.size*.25  * self.best_gmm.weights_[it]
             ax1.plot(x,y_it, color=self.colors[it])
             
        y*=self.dist_mod_chain.size*.25
        ax1.plot(x, y, 'k--', linewidth=1.5)

        
        plt.show()


# class to store clusters in posterior space

class posterior_cluster:

    def __init__(self, data, probs):
        self.data=data
        self.probs=probs
        
        self.set_weight()
        
        print np.mean(self.data[:,1]), np.mean(self.data[:,2]), self.weight
        
    def __len__(self):
        return self.data.shape[0]
        
    def set_weight(self, weight=None):
        if weight:
            self.weight=weight
        else:
            self.weight=np.mean(np.exp(self.probs))*max(np.std(self.data[:,1]),0.01)*max(np.std(self.data[:,2]),0.01)
                     
