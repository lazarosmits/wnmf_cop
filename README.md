# Weighted Non-negative Matrix Factorization

This repository contains code that was used in the analyses described in the paper [Discovering Low-Dimensional Descriptions of Multineuronal Dependencies (Mitskopoulos and Onken, 2023)](https://www.mdpi.com/1099-4300/25/7/1026) 

The code implements a modified version of the classic non-negative matrix factorization (NMF) algorithm [(Lee and Seung, 1999)](https://www.nature.com/articles/44565) by incorporating a weight matrix.In standard NMF, an input data matrix, $$X$$, is factorized into a product of low-dimensional matrices: $$W$$, which holds activation coefficients, and H, which contains low-dimensional features or modules. In this weighted non-negative matrix factorization algorithm (WNMF), the weight matrix, which we can call $$U$$, places varying emphasis on different values of the features/modules in $$H$$.

This weighting approach is especially useful for dimensionality reduction on datasets with overlapping features, such as bivariate (2D) copulas, where standard NMF struggles to separate distinct features related to copula density functions. However, WNMF is versatile and can be applied to other cases where overlapping features require dimensionality reduction, such as image or time series data.

Below are some code snippets to provides an example of WNMF usage with synthetic copula data similar to the setup in Mitskopoulos and Onken (2023). You can view and run the full version in the file `test_cop_wnmf.py`. To construct these synthetic copula data I used the [mixed vines package](https://github.com/asnelt/mixedvines?tab=readme-ov-file), by [Onken and Panzeri (2016)](https://proceedings.neurips.cc/paper_files/paper/2016/hash/fb89705ae6d743bf1e848c206e16a1d7-Abstract.html)


# Feature discovery for 2D copulas with NMF
Suppose you have a dataset and want to use copulas to describe statistical relationships between pairs of its variables. Copulas are statistical tools designed for this purpose, as they capture the shape of these relationships. Specifically, for pairs of variables, we use bivariate (2D) copulas, which represent probability distributions. Here, our object of interest—on which we’ll apply NMF—is a set of 2D density plots.

To illustrate, let’s create a toy dataset with two distinct types of copulas: Frank and Clayton. We’ll generate 20 instances of each type, randomly selecting the `theta` parameter for each copula from a specified range of values. For now, we’ll start with a straightforward case where the tail regions of the copulas (i.e., values in the density plot corners) don’t overlap.

```python
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy import stats
from mixedvines.copula import ClaytonCopula, FrankCopula

from sklearn.decomposition import NMF
from WNMF import wnmf

# generate synthetic samples for Frank and Clayton copulas
n_samp=20000
copula1_samples= FrankCopula(theta=5,rotation='90°').rvs(n_samp)
copula2_samples= ClaytonCopula(theta=5).rvs(n_samp)

# a quick way of estimating probability densities for the samples 
# by gaussian smoothing of the empirical 2D counts
size=100
cop1_hist= np.histogram2d(copula1_samples[:,0],copula1_samples[:,1],
                                     bins=[np.linspace(0,1,size+1),
                                           np.linspace(0,1,size+1)])[0]
cop2_hist= np.histogram2d(copula2_samples[:,0],copula2_samples[:,1],
                                     bins=[np.linspace(0,1,size+1),
                                           np.linspace(0,1,size+1)])[0]
cop_dens1= gaussian_filter(cop1_hist, sigma=4, mode='mirror').flatten()
cop_dens2= gaussian_filter(cop2_hist, sigma=2, mode='mirror').flatten()

```

A schematic of the toy dataset along with the copula densities within looks like this

<img src="https://github.com/user-attachments/assets/e2a90d93-a081-44d9-9333-701a7ab8714d" width="500">


Now let's apply NMF with 2 factors on these data.

```python
# deploy standard NMF with 2 factors
n_fac=2
nmf_cop = NMF(n_components=n_fac)
# extract W coefficients that show which rows of H correspond 
# to which copula module/feature
W = nmf_cop.fit_transform(cop_dens)
# extract H copula modules/features
H = nmf_cop.components_

# plot results
plt.figure()
plt.subplot(2,2,1)
plt.bar(np.arange(W.shape[0]),W[:,0])
plt.title('W coefficients')
plt.ylabel('Factor 1')
plt.subplot(2,2,2)
plt.pcolor(H[0,:].reshape(100,100))
plt.title('H Copula modules')
plt.xticks([0,50,100],[0,0.5,1])
plt.yticks([0,50,100],[0,0.5,1])
plt.subplot(2,2,3)
plt.bar(np.arange(W.shape[0]),W[:,1])
plt.ylabel('Factor 2')
plt.xlabel('Artificial copula index')
plt.subplot(2,2,4)
plt.pcolor(H[1,:].reshape(100,100))
plt.xticks([0,50,100],[0,0.5,1])
plt.yticks([0,50,100],[0,0.5,1])

````

Now let's visualize the results from NMF

<img src="https://github.com/user-attachments/assets/109682db-f351-4577-964d-0de071d13491" width="550">
