import sklearn.mixture as sk_m
import numpy as np

class MeanCov_GMM(sk_m.GMM):
    """ 
    Class to enable GMM fitting starting with mean and covariance
    matrices initialised to those of the data.

    Uses an affine transformation to transform data to 0 mean and
    identity covariance in fit(), then transforms means and
    covariances found back to original coordinate system.
    
    All functions and parameters are the same as sklearn.mixture.GMM
    """     

       
    def fit(self,X):
        
        X_mean = np.mean(X, axis=0)
    
        X_modified = (X - X_mean[np.newaxis,:]) 
        X_covar=np.dot(X_modified.T,X_modified)/(X_modified.shape[0])
        X_covar_L=np.linalg.cholesky(X_covar)
        X_covar_invL=np.linalg.solve(X_covar_L,np.eye(X_covar.shape[0]))        
        X_modified = np.dot(X_covar_invL, X_modified.T).T
        
        self = super(MeanCov_GMM,self).fit(X_modified)
        
        means=self.means_.copy()
        for i in range(self.covars_.shape[0]):
          self.means_[i] = X_mean + np.dot(X_covar_L, means[i])

        covars=self.covars_.copy()
        for i in range(self.covars_.shape[0]):
                self.covars_[i] = np.dot(X_covar_L, np.dot(covars[i], X_covar_L.T) )

        return self            
                
            
