# RegEM: Regularized Expectation Maximization

This repository contains a collection of Matlab modules for 

- the estimation of mean values and covariance matrices from incomplete datasets, and
- the imputation of missing values in incomplete datasets.

The modules implement the regularized EM algorithm described in

T. Schneider, 2001: Analysis of incomplete climate data: Estimation of mean values and covariance matrices and imputation of missing values. Journal of Climate, 14, 853-871. http://dx.doi.org/10.1175/1520-0442(2001)014%3C0853:AOICDE%3E2.0.CO;2

The EM algorithm for Gaussian data is based on iterated linear regression analyses. In the regularized EM algorithm, a regularized estimation method replaces the conditional maximum likelihood estimation of regression parameters in the conventional EM algorithm for Gaussian data. The modules here provide truncated total least squares (with fixed truncation parameter) and ridge regression with generalized cross-validation as regularized estimation methods.

The implementation of the regularized EM algorithm is modular, so that the modules that perform he regularized estimation of regression parameters (e.g., ridge regression and generalized cross-validation) can be exchanged for other regularization methods and other methods of determiningca regularization parameter. Per-Christian Hansen's Regularization Tools contain Matlab modules implementing a collection of regularization methods that can be adapted to fit into the framework of the EM algorithm. The generalized cross-validation modules of the regularized EM algorithm are adapted from Hansen's generalized cross-validation modules.

In the Matlab implementation of the regularized EM algorithm, more emphasis was placed on the modularity of the program code than on computational efficiency. The regularized EM algorithm is currently being developed further under a project funded by the U.S. National Science Foundation's Paleo Perspectives on Climate Change program.


# Installation

The program package consists of several Matlab modules. To install the programs, download the package into a directory that is accessible by Matlab. 

Starting Matlab and invoking Matlab's online help function

help filename

displays information on the module filename.m.

# Module Descriptions

| Module                  | Description                                                                         |
|-------------------------| ----------------------------------------------------------------------------------- |
| CHANGES                 | Recent significant changes of the programs                                          |
| center.m                | Centers data by subtracting the mean                                                |
| gcvfctn.m               | Evaluates generalized cross-validation function (auxiliary to gcvridge.m)           |
| gcvridge.m              | Finds minimum of generalized cross-validation function for ridge regression         |
| iridge.m                | Computes regression parameters by individual ridge regressions                      |
| kcv_ttls.m              | Selects truncation parameter for TTLS by K-fold cross-validation                    |
| kcvindices.m            | Returns random indices for K-fold cross-validation                                  |
| missingness_patterns.m  | Returns unique patterns of missing values in a data matrix                          |
| mridge.m                | Computes regression parameters by a multiple ridge regression                       |
| nancov.m                | Sample covariance matrix of available values in incomplete dataset                  |
| nanmean.m               | Sample mean of available values in incomplete dataset                               |
| nanstd.m                | Standard deviation of available values in incomplete dataset                        |
| nansum.m                | Sum over available values in incomplete dataset                                     |
| pca_truncation_criteria.m | Computes criteria for truncating principal component analyses                     |
| peigs.m                 | Computes positive eigenvalues and corresponding eigenvectors                        |
| pttls.m                 | Computes regression parameters by truncated total least squares                     |
| regem.m                 | Driver module for regularized EM algorithm                                          |
| standardize.m           | Standardizes data by subtracting the mean and scaling with the standard deviation   |
