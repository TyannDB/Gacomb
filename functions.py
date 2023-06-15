import pandas as pd
import numpy as np

def m_corrective_error(nmocks,nvec,nparam):
    """
    nmocks : int : # of mocks used to estimate the covariance
    nvec : int : size of the datavector AFTER applying the cuts
    nparam : int : number of parameters of the model used
    
    return : m1,m2 : int : correctif factors to multiply the cov matrices of the inference parameters 
                           (use m2 when the inference is performed on mocks used to estimate the covariance)
    """
    ##-- Correction facteur Dodelson 2013
    A = 2 / ((nmocks-nvec-1)*(nmocks-nvec-4))
    B = (nmocks-nvec-2) * A/2
    m = 1+B*(nvec-nparam)
    ##-- Correction from Percival 2014
    m1 = m / (1+A+B*(nparam+1))
    m2 = m1/(1 - (nvec + 1.)/(nmocks-1))
    
    return m1,m2

class CGP:
    """
    Combine parameters constraints according to [Sanchez et al 2016]
    """
    def __init__(self,df_config,df_fourier,d_vect_tot,param,cov_config=None,cov_fourier=None,n_methods=2):
        """
        df_config : pandas dataframe : best fit parameters for every mock realisation in configuration space
        df_fourier : pandas dataframe : best fit parameters for every mock realisation in fourier space
        d_vect_tot : list of array : [ [a_para_cs,a_perp_cs], [a_para_fs,a_perp_fs]] each element of the list in an array 
                                    containing the values of the fitted parameters for one method (CS or FS)                        
        param : list : names of the parameters in the df_config/df_fourier
        cov_config : 2D numpy array : covariance of the best fitted parameters in FS. Can be None for no rescaling.
        cov_fourier : 2D numpy array : covariance of the best fitted parameters in FS. Can be None for no rescaling.
        n_methods : int : number of method to combine for now can only be 2 : CS and FS
        """

        self.n_methods = n_methods
        self.nmocks = df_config.shape[0]
        self.param=param
        self.n_param = len(param)
        self.df_config = df_config
        self.df_fourier = df_fourier
        self.d_vect_tot = d_vect_tot
        self.cov_config = cov_config
        self.cov_fourier = cov_fourier

        #-- Compute the covariance matrix between the parameters or the fourier and config method
        self.cov_mocks = self.compute_cov(self.n_param,self.n_methods)
        
        #-- if the covariances from the fit of the parameters for the different methods is given : rescale
        if cov_config is not None and cov_fourier is not None:
            self.cov = self.rescale_cov(self.n_param,self.n_methods,self.cov_mocks,self.cov_config,self.cov_fourier)
        else:
            self.cov = self.cov_mocks 

        self.precision = self.inverse_cov(self.cov,nmocks=self.nmocks)
        #-- Reshape the total precision matrix into a matrix with sub matrix elements
        self.precision_mat = self.reshape_precision(self.precision,self.n_methods,self.n_param)
        #-- Compute the combined covariance matrix of the parameters
        self.cov_comb = self.cov_comb(self.precision_mat)
        #-- Compute the combined data vector
        self.d_comb = self.d_comb(self.d_vect_tot,self.cov_comb,self.precision_mat,self.n_methods)

    def compute_cov(self,n_param,n_methods):
        nmocks = self.nmocks
        n_param=self.n_param
        param = self.param
        mat = np.zeros((nmocks,n_param*n_methods))

        for l in range (nmocks):
            for c,name in zip(range(n_param),param):
                mat[l,c] = self.df_config[name].iloc[l]
                mat[l,c+n_param] = self.df_fourier[name].iloc[l]
        cov_tot = np.cov(mat,rowvar=False)
        return cov_tot
    
    def rescale_cov(self,n_param,n_methods,cov_mocks,cov_config,cov_fourier):

        cov_tot = cov_mocks*1
        #--compute the maximum correlation  : (3,) array for data and for mocks
        err_data_config = np.sqrt(np.diag(cov_config))
        err_data_fourier = np.sqrt(np.diag(cov_fourier))
        err_mocks_config = np.sqrt(np.diag(cov_tot[0:n_param,0:n_param]))
        err_mocks_fourier = np.sqrt(np.diag(cov_tot[n_param:n_param*2,n_param:n_param*2]))

        rho_max_mocks = np.zeros(n_param)
        rho_max_data = np.zeros(n_param)
        for i in range(n_param):
            cd = err_data_config[i]/err_data_fourier[i]
            cm = err_mocks_config[i]/err_mocks_fourier[i]
            rho_max_data[i] = cd if cd <= 1.0 else 1/cd
            rho_max_mocks[i] = cm if cm <= 1.0 else 1/cm

        r = rho_max_data/rho_max_mocks

        #--compute the correlation matrices
        corr_tot = cov_tot/np.sqrt(np.outer(np.diag(cov_tot),np.diag(cov_tot)))
        corr_config = cov_config/np.sqrt(np.outer(np.diag(cov_config),np.diag(cov_config)))
        corr_fourier = cov_fourier/np.sqrt(np.outer(np.diag(cov_fourier),np.diag(cov_fourier)))

        #-- Replace the diag blocks by the correlations given from the fit
        corr_tot[0:n_param,0:n_param]=corr_config
        corr_tot[n_param:n_param*n_methods,n_param:n_param*n_methods]=corr_fourier

        #-- Rescale the diag terms of the off diag blocks 
        for i in range(n_param):
            corr_tot[0:n_param,n_param:n_param*n_methods][i,i] *= r[i]
            corr_tot[n_param:n_param*n_methods,0:n_param][i,i] *= r[i]


        #-- Rescale the off diag terms of the off diag blocks
        for i in range(n_param):
            for j in range(n_param):
                if i!=j:
                    corr_tot[0:n_param,n_param:n_param*n_methods][i,j] = 0.25*(corr_tot[0:n_param,n_param:n_param*n_methods][i,i] + corr_tot[0:n_param,n_param:n_param*n_methods][j,j]) * \
                    (corr_config[i,j] + corr_fourier[i,j])
                    corr_tot[n_param:n_param*n_methods,0:n_param][i,j] = 0.25*(corr_tot[n_param:n_param*n_methods,0:n_param][i,i] + corr_tot[n_param:n_param*n_methods,0:n_param][j,j]) * \
                    (corr_config[i,j] + corr_fourier[i,j])


        #-- rescale the correlation matrix into a covariance with the the diagonal errors from the data
        err = np.concatenate((err_data_config,err_data_fourier),axis=None)
        cov_tot = corr_tot* np.outer(err,err)
        #cov_tot = corr_tot* np.outer(np.sqrt(np.diag(cov_tot)),np.sqrt(np.diag(cov_tot)))

        return cov_tot

    def inverse_cov(self,cov, nmocks=0):
        inv_cov = np.linalg.inv(cov)
        if nmocks > 0: #--Hartlap correction
            correction = (1 - (cov.shape[0] + 1.)/(nmocks-1))
            inv_cov *= correction
        return inv_cov

    def reshape_precision(self,precision,n_methods,n_param):
        precision_mat = np.zeros((n_methods,n_methods),dtype='object')
        for i in range(n_methods):
            for j in range(n_methods):
                precision_mat[i,j] = np.array(precision[i*n_param:(i+1)*n_param,j*n_param:(j+1)*n_param])

        return precision_mat

    def cov_comb(self,precision_mat):
        precision_comb = np.sum(precision_mat)
        cov_comb = np.linalg.inv(precision_comb)
        return cov_comb

    def d_comb(self,d_vect_tot,cov_comb,precision_mat,n_methods):
        a=0
        for i in range(n_methods):
            a +=np.sum(precision_mat,axis=0)[i]@ d_vect_tot[i]
        return cov_comb @ a

    
    
    
    
