import pandas as pd
import numpy as np
from functions import CGP

#--parameters best fitted on a bunch of mocks
df_EZ_prerecon_CS = pd.read_csv('CS_mock_bestvalues.csv') 
df_EZ_prerecon_FS = pd.read_csv('FS_mock_bestvalues.csv') 

param = ['alpha_perp','alpha_para']
d_vect_tot = [np.array([1.05,0.88]),
              np.array([1.10,0.91])]

c = CGP(df_EZ_prerecon_CS,df_EZ_prerecon_FS,d_vect_tot,param,cov_config=None,cov_fourier=None)
cov_comb = c.cov_comb
d_comb = c.d_comb
print('best value : ',d_comb)
print('Error : ',np.sqrt(np.diag(cov_comb)))


##### If you have parameters covariances for each methods

cov_config = np.array([[ 7.29430578e-05, -7.04838572e-05],
       [-7.04838572e-05,  3.48195818e-04]])
cov_fourier = np.array([[ 7.29430578e-05, -7.04838572e-05],
       [-7.04838572e-05,  3.48195818e-04]])

c = CGP(df_EZ_prerecon_CS,df_EZ_prerecon_FS,d_vect_tot,param,cov_config=cov_config,cov_fourier=cov_fourier)
cov_comb = c.cov_comb
d_comb = c.d_comb
print('best value : ',d_comb)
print('Error : ',np.sqrt(np.diag(cov_comb)))

