# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:49:15 2024

@author: lazar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:02:46 2022

@author: lazar
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from scipy.ndimage import gaussian_filter
from scipy import stats
from WNMF import wnmf

from mixedvines.copula import ClaytonCopula, FrankCopula


#%% Copula densities Clayton and Frank

size=100
n_samp=20000
copula1_samples= FrankCopula(theta=5,rotation='90°').rvs(n_samp)
copula2_samples= ClaytonCopula(theta=5).rvs(n_samp)


cop1_hist= np.histogram2d(copula1_samples[:,0],copula1_samples[:,1],
                                     bins=[np.linspace(0,1,size+1),
                                           np.linspace(0,1,size+1)])[0]
cop2_hist= np.histogram2d(copula2_samples[:,0],copula2_samples[:,1],
                                     bins=[np.linspace(0,1,size+1),
                                           np.linspace(0,1,size+1)])[0]
cop_dens1= gaussian_filter(cop1_hist, sigma=4, mode='mirror').flatten()
cop_dens2= gaussian_filter(cop2_hist, sigma=2, mode='mirror').flatten()

plt.figure()
plt.subplot(1,2,1)
plt.pcolor(cop_dens1.reshape(size,size))
plt.title('Symmetric, light tailed')
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.pcolor(cop_dens2.reshape(size,size))
plt.title('Asymmetric, heavy tailed')
plt.xticks([])
plt.yticks([])


#%% construct matrix with artificial copulas, non-overlapping tails

n_reps=20

f_theta=np.random.randint(4,6,n_reps)
c_theta=np.random.randint(4,6,n_reps)
       
# create artificial copula data to test nmf
xx, yy = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
points= np.stack((xx.flatten(),yy.flatten())).T

frank_samps=np.zeros((len(f_theta),n_samp,2))
cl_samps=np.zeros((len(c_theta),n_samp,2))
cop_dens=np.zeros((int(n_reps*2),points.shape[0]))

for i in range(n_reps):
    f_cop=FrankCopula(theta=f_theta[i],rotation='90°')
    frank_samps[i,:,:]=f_cop.rvs(size=n_samp)
    cop_hist= np.histogram2d(frank_samps[i,:,0],frank_samps[i,:,1],
                             bins=[np.linspace(0,1,size+1),
                                   np.linspace(0,1,size+1)])[0]
    cop_dens[i,:]= gaussian_filter(cop_hist, sigma=2, mode='mirror').flatten()
    cop_dens[i,:]=cop_dens[i,:]/np.sum(cop_dens[i,:])
    
    if i<len(c_theta):
        
        c_cop=ClaytonCopula(theta=c_theta[i])
        cl_samps[i,:,:]=c_cop.rvs(size=n_samp)
        cop_hist= np.histogram2d(cl_samps[i,:,0],cl_samps[i,:,1],
                                  bins=[np.linspace(0,1,size+1),
                                        np.linspace(0,1,size+1)])[0]
        cop_dens[i+n_reps,:]= gaussian_filter(cop_hist, sigma=2, mode='mirror').flatten()
        cop_dens[i+n_reps,:]=cop_dens[i+n_reps,:]/np.sum(cop_dens[i+n_reps,:])

#%% Unweighted NMF can extract true copulas when tails are not overlapping

n_fac=2
nmf_cop = NMF(n_components=n_fac)
W = nmf_cop.fit_transform(cop_dens)
H = nmf_cop.components_

# plot results
plt.figure()
plt.subplot(2,2,1)
plt.bar(np.arange(W.shape[0]),W[:,0])
plt.title('Activation coefficients')
plt.ylabel('Factor 1')
plt.subplot(2,2,2)
plt.pcolor(H[0,:].reshape(100,100))
plt.title('Copula modules')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
plt.bar(np.arange(W.shape[0]),W[:,1])
plt.ylabel('Factor 2')
plt.xlabel('Artificial copula index')
plt.subplot(2,2,4)
plt.pcolor(H[1,:].reshape(100,100))
plt.xticks([])
plt.yticks([])

#%% construct matrix with artificial copulas, overlapping tails
       
# create artificial copula data to test nmf
xx, yy = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
points= np.stack((xx.flatten(),yy.flatten())).T

frank_samps=np.zeros((len(f_theta),n_samp,2))
cl_samps=np.zeros((len(c_theta),n_samp,2))
cop_dens=np.zeros((int(n_reps*2),points.shape[0]))

for i in range(n_reps):
    f_cop=FrankCopula(theta=f_theta[i])
    frank_samps[i,:,:]=f_cop.rvs(size=n_samp)
    cop_hist= np.histogram2d(frank_samps[i,:,0],frank_samps[i,:,1],
                             bins=[np.linspace(0,1,size+1),
                                   np.linspace(0,1,size+1)])[0]
    cop_dens[i,:]= gaussian_filter(cop_hist, sigma=2, mode='mirror').flatten()
    cop_dens[i,:]=cop_dens[i,:]/np.sum(cop_dens[i,:])
    
    if i<len(c_theta):
        
        c_cop=ClaytonCopula(theta=c_theta[i])
        cl_samps[i,:,:]=c_cop.rvs(size=n_samp)
        cop_hist= np.histogram2d(cl_samps[i,:,0],cl_samps[i,:,1],
                                  bins=[np.linspace(0,1,size+1),
                                        np.linspace(0,1,size+1)])[0]
        cop_dens[i+n_reps,:]= gaussian_filter(cop_hist, sigma=2, mode='mirror').flatten()
        cop_dens[i+n_reps,:]=cop_dens[i+n_reps,:]/np.sum(cop_dens[i+n_reps,:])
        
#%% Unweighted NMF fails to extract true copulas for overlapping tails

n_fac=2
nmf_cop = NMF(n_components=n_fac)
W = nmf_cop.fit_transform(cop_dens)
H = nmf_cop.components_

# plot results
plt.figure()
plt.subplot(2,2,1)
plt.bar(np.arange(W.shape[0]),W[:,0])
plt.title('Activation coefficients')
plt.ylabel('Factor 1')
plt.subplot(2,2,2)
plt.pcolor(H[0,:].reshape(100,100))
plt.title('Copula modules')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
plt.bar(np.arange(W.shape[0]),W[:,1])
plt.ylabel('Factor 2')
plt.xlabel('Artificial copula index')
plt.subplot(2,2,4)
plt.pcolor(H[1,:].reshape(100,100))
plt.xticks([])
plt.yticks([])

#%% Construct weight matrices by margin icdf


# create two hypothetical poisson neuron margins 
margin_1_icdf=stats.poisson.ppf(
    np.linspace(0.001,0.99,size),mu=5,loc=1
                     )-np.linspace(0.001,0.99,size)

margin_2_icdf=stats.poisson.ppf(
    np.linspace(0.001,0.99,size),mu=7,loc=1
                     )-np.linspace(0.001,0.99,size)


# take outer product of the the two margin icdfs
oprod=gaussian_filter(margin_1_icdf.reshape(-1,1)@margin_2_icdf.reshape(1,-1), 
                      sigma=11)**4
# concatenate rotated versions of the outer product to cover all corners
rot_op=oprod+np.rot90(oprod)+np.rot90(np.rot90(oprod))+np.rot90(np.rot90(np.rot90(oprod)))
rot_op=rot_op/np.mean(rot_op)
# visualize the weighting matrix
plt.figure()
plt.pcolor(rot_op)
plt.yticks([])
plt.xticks([])

# tile the matrices together for all the data
err_mat=np.tile(rot_op, cop_dens.shape[0]).reshape(cop_dens.shape[0],-1)

#%% Deploy Weighted NMF
W,H,err_ao=wnmf(cop_dens.T,k=n_fac,
                          Wght=err_mat.T)

# plot results
plt.figure()
plt.subplot(2,2,1)
plt.bar(np.arange(W.shape[0]),W[:,0])
plt.title('Activation coefficients')
plt.ylabel('Factor 1')
plt.subplot(2,2,2)
plt.pcolor(H[0,:].reshape(100,100))
plt.title('Copula modules')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
plt.bar(np.arange(W.shape[0]),W[:,1])
plt.ylabel('Factor 2')
plt.xlabel('Artificial copula index')
plt.subplot(2,2,4)
plt.pcolor(H[1,:].reshape(100,100))
plt.xticks([])
plt.yticks([])

#%% Deploy Weighted NMF with L1 regularization on W

W,H,err_ao=wnmf(cop_dens.T,k=n_fac,
                          Wght=err_mat.T,l1_reg_W=0.5)

# plot results
plt.figure()
plt.subplot(2,2,1)
plt.bar(np.arange(W.shape[0]),W[:,0])
plt.title('Activation coefficients')
plt.ylabel('Factor 1')
plt.subplot(2,2,2)
plt.pcolor(H[0,:].reshape(100,100))
plt.title('Copula modules')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
plt.bar(np.arange(W.shape[0]),W[:,1])
plt.ylabel('Factor 2')
plt.xlabel('Artificial copula index')
plt.subplot(2,2,4)
plt.pcolor(H[1,:].reshape(100,100))
plt.xticks([])
plt.yticks([])
