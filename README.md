# Weighted Non-negative Matrix Factorization

This repository contains code that was used in the analyses described in the paper [Discovering Low-Dimensional Descriptions of Multineuronal Dependencies (Mitskopoulos and Onken, 2023)](https://www.mdpi.com/1099-4300/25/7/1026) 

The code implements a modified version of the classic non-negative matrix factorization (NMF) algorithm [(Lee and Seung, 1999)](https://www.nature.com/articles/44565) by incorporating a weight matrix.In standard NMF, an input data matrix, $$X$$, is factorized into a product of low-dimensional matrices: $$W$$, which holds activation coefficients, and H, which contains low-dimensional features or modules. In this weighted non-negative matrix factorization algorithm (WNMF), the weight matrix, which we can call $$U$$, places varying emphasis on different values of the features/modules in $$H$$.

This weighting approach is especially useful for dimensionality reduction on datasets with overlapping features, such as bivariate (2D) copulas, where standard NMF struggles to separate distinct features related to copula density functions. However, WNMF is versatile and can be applied to other cases where overlapping features require dimensionality reduction, such as image or time series data.

The code below provides an example of WNMF usage with synthetic copula data similar to the setup in Mitskopoulos and Onken (2023). To construct these synthetic copula data I used the [mixed vines package](https://github.com/asnelt/mixedvines?tab=readme-ov-file), by [Onken and Panzeri (2016)](https://proceedings.neurips.cc/paper_files/paper/2016/hash/fb89705ae6d743bf1e848c206e16a1d7-Abstract.html)


