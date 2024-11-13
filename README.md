# Weighted Non-negative Matrix Factorization

This repository contains code that was used in the analyses described in the paper [Discovering Low-Dimensional Descriptions of Multineuronal Dependencies (Mitskopoulos and Onken, 2023)](https://www.mdpi.com/1099-4300/25/7/1026) 

The code implements a modified version of the classic non-negative matrix factorization (NMF) algorithm [(Lee and Seung, 1999)](https://www.nature.com/articles/44565) by incorporating a weight matrix.In standard NMF, an input data matrix, $$X$$, is factorized into a product of low-dimensional matrices: $$W$$, which holds activation coefficients, and H, which contains low-dimensional features or modules. In this weighted non-negative matrix factorization algorithm (WNMF), the weight matrix, which we can call $$U$$, places varying emphasis on different values of the features/modules in $$H$$.

This weighting approach is especially useful for dimensionality reduction on datasets with overlapping features, such as bivariate (2D) copulas, where standard NMF struggles to separate distinct features related to copula density functions. However, WNMF is versatile and can be applied to other cases where overlapping features require dimensionality reduction, such as image or time series data.

Below are some code snippets to provides an example of WNMF usage with synthetic copula data similar to the setup in Mitskopoulos and Onken (2023). You can view and run the full version in the file `test_cop_wnmf.py`. To construct these synthetic copula data I used the [mixed vines package](https://github.com/asnelt/mixedvines?tab=readme-ov-file), by [Onken and Panzeri (2016)](https://proceedings.neurips.cc/paper_files/paper/2016/hash/fb89705ae6d743bf1e848c206e16a1d7-Abstract.html)


# Feature discovery for 2D copulas with NMF
Suppose you have a dataset and want to use copulas to describe statistical relationships between pairs of its variables. Copulas are statistical tools designed for this purpose, as they capture the shape of these relationships. Specifically, for pairs of variables, we use bivariate (2D) copulas, which represent probability distributions. Here, our object of interest—on which we’ll apply NMF—is a set of 2D density plots. The rationale for applying NMF is that, in large datasets, we often want to distill the most important features that might be hard to discern among numerous data points.

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

<img src="https://github.com/user-attachments/assets/7a839014-d87b-4299-a624-f86d4a052b29" width="500">


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

As is obvious in the following visualization, NMF does a good job of discovering the copula shapes

<img src="https://github.com/user-attachments/assets/109682db-f351-4577-964d-0de071d13491" width="550">


# Feature discovery for 2D copulas with overlapping tail regions

In the previous example, I deliberately rotated the Frank copulas by 90° so that the shapes do not overlap in the tail regions, which is where a lot of probability mass is concentrated for the Clayton copulas. Things are not as easy, however, if the copula shapes in question have significant overlap as is the case in the following scenario. 

<img src="https://github.com/user-attachments/assets/bec6fdbb-41ca-4e8b-b55a-aac35805415b" width="550">

Cases like this can be quite common in real data. If we apply standard NMF in this case, the outcome can be seen in the following figure. The features of the shapes are not well separated in the `H` copula modules/factors.

<img src="https://github.com/user-attachments/assets/b9f5574b-9b6a-4920-8060-b40faa14b2bc" width="550">

# Adding a weight matrix can improve NMF feature discovery for overlapping shapes

The most major issue seems to be in the tail regions, which is a very important part for copula-based approaches. After all, copulas are a tool that is very useful for statistical dependencies with heavy tails, that is when the variables are more strongly co-dependent in extreme values. So if in a dataset we encounter both heavy tail dependencies (Clayton-like copulas) and dependencies with lighter tails (Frank-like copulas) we need a way to address this issue in order to achieve better results. 

One way is to introduce a weight matrix that forces the algorithm to place more emphasis on representing the tail regions accurately. This weight matrix can be incorporated in multiplicative update rules for `W` and `H`. But how does one construct such a matrix? One option is by making use of the probability distributions of the variables themselves, and specifically the inverse cumulative probability functions (ICDF). The ICDFs can transform samples from the copula space to the range of values observed in the data. For every pair of variables correspoding to a 2D copula one can take the outer product or the ICDFs of these variables. This results in a matrix the values of which are highest where the tails of the variables' distributions coincide, depending on how these distributions are skewed. The final step is to take a summation of this outer product with rotated versions of it, in order to place equal focus on all four corners of the copulas. The final result is the weight matrix we want. The code below illustrates how to achieve that for hypothetical poisson distributions, since we are working with synthetic data in this case.

```python
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
plt.xticks([0,50,100],[0,0.5,1])
plt.yticks([0,50,100],[0,0.5,1])
plt.title('ICDF Weight matrix')

# tile the matrices together for all the data
err_mat=np.tile(rot_op, cop_dens.shape[0]).reshape(cop_dens.shape[0],-1)
```

Let's see how the weight matrix looks like.

<img src="https://github.com/user-attachments/assets/5584ba26-7fb9-407b-8edf-28e3952e70a9" width="350">







