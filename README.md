# Weighted Non-negative Matrix Factorization

This repository contains code that was used in the analyses described in the paper [Discovering Low-Dimensional Descriptions of Multineuronal Dependencies](https://www.mdpi.com/1099-4300/25/7/1026) 
The code implements a classic non-negative matrix factorization (NMF) algorithm with random initialization and multiplicative updates. In addition, it incorporates a weighting matrix, allowing the reconstruction error to be penalized differently across the data points. This approach was developed to apply dimensionality reduction on a dataset of bivariate copulas, where overlapping features made it challenging for traditional NMF to effectively separate shared structures.
