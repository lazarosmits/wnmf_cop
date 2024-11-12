# Weighted Non-negative Matrix Factorization

This repository contains code that was used in the analyses described in the paper [Discovering Low-Dimensional Descriptions of Multineuronal Dependencies (Mitskopoulos and Onken, 2023)](https://www.mdpi.com/1099-4300/25/7/1026) 

The code implements a modified version of the classic non-negative matrix factorization (NMF) algorithm [(Lee and Seung, 1999)](https://www.nature.com/articles/44565) by incorporating a weight matrix. In standard NMF an input data matrix X is factorised into a product of low dimensional matrices W, containing activation coefficients and H, containing low dimensional features/modules. In this weighted non-negative matrix factorization algorithm (WNMF), the weight matrix, which we can call U places different levels of emphasis on different values of the features/modules in H. This was a useful approach for applying dimensionality reduction on a dataset of bivariate (2D) copulas, where overlapping features made it challenging for standard NMF to effectively separate distinct features corresponding the copula density functions. However, WNMF can be used for any scenario where one wants to apply dimensionality reduction to extract features that are overlapping in the data, e.g. in image data or time series data.

The code below provides an example of usage for WNMF with synthetic copula data similar to the example described in Mitskopoulos and Onken (2023). To construct these synthetic copula data I used the [mixed vines package](https://github.com/asnelt/mixedvines?tab=readme-ov-file), from [Onken and Panzeri (2016)](https://proceedings.neurips.cc/paper_files/paper/2016/hash/fb89705ae6d743bf1e848c206e16a1d7-Abstract.html)


