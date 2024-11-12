# Weighted Non-negative Matrix Factorization

This repository contains code that was used in the analyses described in the paper [Discovering Low-Dimensional Descriptions of Multineuronal Dependencies (Mitskopoulos and Onken, 2023)](https://www.mdpi.com/1099-4300/25/7/1026) 

The code implements a modified version of the classic non-negative matrix factorization (NMF) algorithm [(Lee and Seung, 1999)](https://www.nature.com/articles/44565) by incorporating a weight matrix.In standard NMF, an input data matrix, $$X$$, is factorized into a product of low-dimensional matrices: $$W$$, which holds activation coefficients, and H, which contains low-dimensional features or modules. In this weighted non-negative matrix factorization algorithm (WNMF), the weight matrix, which we can call $$U$$, places varying emphasis on different values of the features/modules in $$H$$.

This weighting approach is especially useful for dimensionality reduction on datasets with overlapping features, such as bivariate (2D) copulas, where standard NMF struggles to separate distinct features related to copula density functions. However, WNMF is versatile and can be applied to other cases where overlapping features require dimensionality reduction, such as image or time series data.

The code below provides an example of WNMF usage with synthetic copula data similar to the setup in Mitskopoulos and Onken (2023). To construct these synthetic copula data I used the [mixed vines package](https://github.com/asnelt/mixedvines?tab=readme-ov-file), by [Onken and Panzeri (2016)](https://proceedings.neurips.cc/paper_files/paper/2016/hash/fb89705ae6d743bf1e848c206e16a1d7-Abstract.html)


# Feature discovery for 2D copulas with NMF
Suppose you are given a dataset and you want to deploy copulas to describe the statistical relationships between pairs of the variables in that dataset. Copulas are statistical tools which are very useful for exactly that purpose, capturing the shape of these statistical relationships. For pairs of variables we have bivariate, that is 2D copulas, and these copulas are essentially probability distributions, so the object of interest here which NMF is applied to is 2D density plots.

Let's make a toy dataset consisting of 2 distinct types of copulas, called Frank and Clayton copulas. And let's make 20 instances for each of the 2 types, where in each of the instances the value of the theta parameter in Frank and Clayton copulas is picked randomly from a range of values. We are going to start with the easy case where the tail regions in the copulas, i.e. the values in the corners of the density, don't overlap.

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

<img src="https://github.com/user-attachments/assets/e2a90d93-a081-44d9-9333-701a7ab8714d" width="48">
