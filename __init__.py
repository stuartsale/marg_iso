# Import stuff

import numpy as np
import iso_lib as il
import sklearn as sk
import sklearn.cluster as sk_c
import sklearn.mixture as sk_m
import matplotlib as mpl
import matplotlib.pyplot as plt
import emcee
import math

from scipy import linalg
from gmm_extra import MeanCov_GMM



def emcee_prob(params, star):
            
    try:
        iso_obj=star.isochrones.query(params[0], params[1], params[2])
    except IndexError:
        return -np.inf+np.zeros(params.shape[1])
        
    R_out_of_bounds=np.logical_or(params[5]<2.05, params[5]>5.05)
    params[5,R_out_of_bounds]=3.1
    
    try:
        a_out_of_bounds=params[4]<-5
        params[4,a_out_of_bounds]=-5
        
        A=np.exp(params[4])
        dist=np.power(10., params[3]/5.+1.)    
        R_gal=np.sqrt( 8000*8000+np.power(dist*star.cosb,2)-2*8000*dist*star.cosb*star.cosl )
        prob= np.log(iso_obj.Jac) -2.3*np.log(iso_obj.Mi) + 2.3026*iso_obj.logage  \
                -np.power(params[0]+(R_gal-8000.)*0.00006,2)/(2*0.04) 
                
        for band in star.mag.keys():
            prob-= np.power(star.mag[band]-(iso_obj.abs_mag[band]+params[3]+iso_obj.AX(band, A, params[5]) ) 
                    ,2)/(2*star.d_mag[band]*star.d_mag[band])

        prob[iso_obj.Jac==0]=-np.inf
        prob[np.logical_or(params[5]<2.05, params[5]>5.05)]=-np.inf
        prob[params[4]<-5]=-np.inf
        
        all_out_of_bounds=np.logical_or(R_out_of_bounds, a_out_of_bounds)
        prob[all_out_of_bounds]=-np.inf
        
        return prob
        
    except OverflowError:
        return -np.inf
                
               

# Class to contain star's data, chain, etc            

class star_posterior:

    # init function
    
    def __init__(self, l, b, mag_in, d_mag_in, isochrones=None, isochrone_file=None, init_bands=None):

    
        self.colors = np.array([x for x in 'bgrcmybgrcmybgrcmybgrcmy'])
        self.colors = np.hstack([self.colors] * 20)
        
        self.l=np.radians(l)
        self.b=np.radians(b)

        self.sinl=np.sin(self.l)
        self.cosl=np.cos(self.l)
        self.sinb=np.sin(self.b)
        self.cosb=np.cos(self.b)
    
        self.mag=mag_in
        self.d_mag=d_mag_in
        
        if init_bands is None:
            self.init_bands=self.mag.keys()[:2]
        else:
            self.init_bands=init_bands
        
        if isochrones is None:
            if isochrone_file is not None:
                isochrones=il.iso_grid_tefflogg(isochrone_file, bands=self.mag.keys())
            else:
                raise IOError("Either an isochrone must be provided or a filename for one given")
        self.isochrones=isochrones
            
        self.MCMC_run=False 
        self.best_gmm=None       
        
                  
                    
    # ==============================================================
    # Functions to work with emcee sampler
    
    def emcee_init(self, N_walkers, chain_length):
    
        self.start_params=np.zeros([N_walkers,6])
    
        guess_set=[]
        guess_set.append([0.,3.663 ,4.57 ,0.,0.,3.1]);	#K4V
        guess_set.append([0.,3.672 ,4.56 ,0.,0.,3.1]);	#K3V
        guess_set.append([0.,3.686 ,4.55 ,0.,0.,3.1]);	#K2V
        guess_set.append([0.,3.695 ,4.55 ,0.,0.,3.1]);	#K1V
        guess_set.append([0.,3.703 ,4.57 ,0.,0.,3.1]);	#K0V
        guess_set.append([0.,3.720 ,4.55 ,0.,0.,3.1]);	#G8V
        guess_set.append([0.,3.740 ,4.49 ,0.,0.,3.1]);	#G5V
        guess_set.append([0.,3.763 ,4.40 ,0.,0.,3.1]);	#G2V
        guess_set.append([0.,3.774 ,4.39 ,0.,0.,3.1]);	#G0V
        guess_set.append([0.,3.789 ,4.35 ,0.,0.,3.1]);	#F8V
        guess_set.append([0.,3.813 ,4.34 ,0.,0.,3.1]);	#F5V
        guess_set.append([0.,3.845 ,4.26 ,0.,0.,3.1]);	#F2V
        guess_set.append([0.,3.863 ,4.28 ,0.,0.,3.1]);	#F0V
        guess_set.append([0.,3.903 ,4.26 ,0.,0.,3.1]);	#A7V
        guess_set.append([0.,3.924 ,4.22 ,0.,0.,3.1]);	#A5V
        guess_set.append([0.,3.949 ,4.20 ,0.,0.,3.1]);	#A3V
        guess_set.append([0.,3.961 ,4.16 ,0.,0.,3.1]);	#A2V
        
#        guess_set.append([0.,3.763 ,3.20 ,0.,0.,3.1]);	#G2III
#        guess_set.append([0.,3.700 ,2.75 ,0.,0.,3.1]);	#G8III
#        guess_set.append([0.,3.663 ,2.52 ,0.,0.,3.1]);	#K1III
#        guess_set.append([0.,3.602 ,1.25 ,0.,0.,3.1]);	#K5III
#        guess_set.append([0.,3.591 ,1.10 ,0.,0.,3.1]);	#M0III

        guess_set.append([0.,3.760 ,4.00 ,0.,0.,3.1]);	#horizontal branch
        guess_set.append([0.,3.720 ,3.80 ,0.,0.,3.1]);	#
        guess_set.append([0.,3.680 ,3.00 ,0.,0.,3.1]);	#
        guess_set.append([0.,3.700 ,2.75 ,0.,0.,3.1]);	#G8III
        guess_set.append([0.,3.680 ,2.45 ,0.,0.,3.1]);	#
        guess_set.append([0.,3.600 ,1.20 ,0.,0.,3.1]);	#K5III
        guess_set.append([0.,3.580 ,0.30 ,0.,0.,3.1]);	#K5III
        
        guess_set=np.array(guess_set)
        
        iso_objs=self.isochrones.query(guess_set[:,0], guess_set[:,1], guess_set[:,2])
        guess_set[:,4]=np.log( ((self.mag[self.init_bands[0]]-self.mag[self.init_bands[1]])
                            -(iso_objs.abs_mag[self.init_bands[0]]-iso_objs.abs_mag[self.init_bands[1]])) 
                            /(iso_objs.AX1[self.init_bands[0]][np.arange(guess_set.shape[0]),11]-iso_objs.AX1[self.init_bands[1]][np.arange(guess_set.shape[0]),11]) )
        guess_set[:,3]=self.mag[self.init_bands[0]] - (iso_objs.AX1[self.init_bands[0]][np.arange(guess_set.shape[0]),11]*guess_set[:,4]
                            +iso_objs.AX2[self.init_bands[0]][np.arange(guess_set.shape[0]),11]*guess_set[:,4]*guess_set[:,4]+iso_objs.abs_mag[self.init_bands[0]])
    
        metal_min=sorted(self.isochrones.metal_dict.keys())[0]
        metal_max=sorted(self.isochrones.metal_dict.keys())[-1]
            
        for it in range(N_walkers):
            self.start_params[it,:]=guess_set[int(np.random.uniform()*len(guess_set))]
            self.start_params[it,0]=metal_min+(metal_max-metal_min)*np.random.uniform()
            self.start_params[it,5]=2.9+0.4*np.random.uniform()
            
        self.Teff_chain=np.zeros(chain_length)
        self.logg_chain=np.zeros(chain_length)
        self.feh_chain=np.zeros(chain_length)
        self.dist_mod_chain=np.zeros(chain_length)
        self.logA_chain=np.zeros(chain_length)
        self.RV_chain=np.zeros(chain_length)        

        self.prob_chain=np.zeros(chain_length)
        self.prior_chain=np.zeros(chain_length)
        self.Jac_chain=np.zeros(chain_length)
        self.accept_chain=np.zeros(chain_length)

        self.photom_chain={}
        for band in self.mag:
            self.photom_chain[band]=np.zeros(chain_length)
        
        self.itnum_chain=np.zeros(chain_length)            
            
            
    def emcee_run(self, iterations=10000, thin=10, burn_in=2000, N_walkers=50, prune=True, prune_plot=False):
    
        self.emcee_init(N_walkers, (iterations-burn_in)/thin*N_walkers)
    
        sampler = emcee.EnsembleSampler(N_walkers, 6, emcee_prob, args=[self])
        
        pos, last_prob, state = sampler.run_mcmc(self.start_params, burn_in)     # Burn-in
        sampler.reset()
 
        if prune:        
            dbscan = sk_c.DBSCAN(eps=0.05)
            pos, last_prob, state = sampler.run_mcmc(pos, 100, rstate0=state, lnprob0=last_prob)     # pruning set
            dbscan.fit(sampler.flatchain[:,1:2])
            labels=dbscan.labels_.astype(np.int)
            
            if prune_plot:

                fig=plt.figure()
                ax1=fig.add_subplot(221)
                
                
                ax1.scatter(sampler.flatchain[:,1], sampler.flatchain[:,2], color='0.5',s=1)
                ax1.scatter(pos[:,1], pos[:,2], color=self.colors[labels].tolist(),s=3)  
                ax1.set_xlim(right=3.5, left=4.5)                 
                ax1.set_ylim(bottom=5., top=2.)
                
                ax1=fig.add_subplot(223)
                ax1.scatter(sampler.flatchain[:,1], sampler.flatlnprobability, color='0.5',s=1)
                ax1.scatter(pos[:,1], last_prob, color=self.colors[labels].tolist(),s=3) 
                ax1.set_xlim(right=3.5, left=4.5)                                   

            
            mean_ln_prob=np.mean(sampler.flatlnprobability)
            cl_list=[]
            weights_list=[]
            weights_sum=0
            for cl_it in range(np.max(labels)+1):
                cl_list.append(posterior_cluster(sampler.flatchain[labels==cl_it,:], sampler.flatlnprobability[labels==cl_it]-mean_ln_prob))
                weights_sum+= cl_list[-1].weight
                weights_list.append(cl_list[-1].weight)
            
            for i in range(N_walkers):
                cluster=np.random.choice(np.max(labels)+1, p=weights_list/np.sum(weights_list))
                index=int( np.random.uniform()*len(cl_list[cluster]) )
                pos[i,:]=cl_list[cluster].data[index,:]
                
            if prune_plot:       
                ax1=fig.add_subplot(222)
                
                
                ax1.scatter(sampler.flatchain[:,1], sampler.flatchain[:,2], color='0.5',s=1)
                ax1.scatter(pos[:,1], pos[:,2], color=self.colors[labels].tolist(),s=3) 
                ax1.set_xlim(right=3.5, left=4.5)                 
                ax1.set_ylim(bottom=5., top=2.)                               
                
                ax1=fig.add_subplot(224)
                ax1.scatter(sampler.flatchain[:,1], sampler.flatlnprobability, color='0.5',s=1)
                ax1.scatter(pos[:,1], last_prob, color=self.colors[labels].tolist(),s=3) 
                ax1.set_xlim(right=3.5, left=4.5)                
                
                plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.6)                
                plt.savefig("prune.pdf")
            
            sampler.reset()

        
        for i, (pos, prob, rstate) in enumerate(sampler.sample(self.start_params, iterations=(iterations-burn_in), storechain=False)):      # proper run
        
            if i%thin==0 and i!=0:
            
                self.feh_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,0]
                self.Teff_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,1]
                self.logg_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,2]
                self.dist_mod_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,3]
                self.logA_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,4]
                self.RV_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=pos[:,5]                
                
                self.prob_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]= prob
                
                self.itnum_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=  i
                
                self.accept_chain[i/thin*N_walkers:(i/thin+1)*N_walkers]=(sampler.acceptance_fraction*i - 
                                                                    self.accept_chain[(i/thin-1)*N_walkers:(i/thin)*N_walkers] )
                
                iso_obj=self.isochrones.query(pos[:,0], pos[:,1], pos[:,2])
                A=np.exp(pos[:,4])
                
                for band in self.mag:
                    self.photom_chain[band][i/thin*N_walkers:(i/thin+1)*N_walkers]=iso_obj.abs_mag[band]
                        

                        
        self.MCMC_run=True
                        
    # ==============================================================                        
    # Fit Gaussians
    
    def gmm_fit(self, max_components=10):
    
        if self.MCMC_run:
            fit_points=np.array([self.dist_mod_chain, self.logA_chain, self.RV_chain]).T

            best_bic=+np.infty
            for n_components in range(1,max_components+1):
                gmm = MeanCov_GMM(n_components=n_components, covariance_type='full',min_covar=0.0001)
                gmm.fit(fit_points)
                if gmm.bic(fit_points)<best_bic-10:
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
            
            self.gmm_sample=self.best_gmm.means_[components]  
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
        X=[self.itnum_chain, self.Teff_chain, self.logg_chain, self.feh_chain, self.dist_mod_chain, self.logA_chain, 
                                self.RV_chain, self.prob_chain, self.prior_chain, self.Jac_chain, self.accept_chain]
        header_txt="N\tTeff\tlogg\tfeh\tdist_mod\tlogA\tRV\tlike\tprior\tJac\taccept"
        for band in self.photom_chain:
            X.append(self.photom_chain[band])
            header_txt+="\t{}".format(band)
        X=np.array(X).T
        header_txt+="\n"

        np.savetxt(filename, X, header=header_txt )
    

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
            ell = mpl.patches.Ellipse(self.best_gmm.means_[it], v[0], v[1], 180 + angle, ec=self.colors[it], fc='none', lw=3)
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

       
    def __len__(self):
        return self.data.shape[0]
        
    def set_weight(self, weight=None):
        if weight:
            self.weight=weight
        else:
            self.weight=np.mean(np.exp(self.probs))*max(np.std(self.data[:,1]),0.01)*max(np.std(self.data[:,2]),0.01)
                     
