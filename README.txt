# GTsurvival

This repository contains python and R implementation of the algorithms proposed in GTsurvival.

## Description

This study introduces GTsurvival, a novel network architecture that combines graph convolutional networks (GCN) with a neural decision tree for Alzheimer’s disease (AD) prediction. GTsurvival utilizes restricted mean survival time (RMST) as pseudo-observations and
directly connects them with baseline variables. By simultaneously predicting RMST at multiple time points, GTsurvival simplifies complex survival analysis into a standard regression problem. Through the joint simulation of RMST, GTsurvival
can effectively utilize shared information and enhance its predictive ability for patients’future survival status.

## Requirements

It is required to install the following dependencies in order to be able to run the code

- [Anaconda3](https://www.anaconda.com/products/individual)
- [python 3](https://www.python.org/downloads/)
- [sklearn](https://pypi.org/project/sklearn/0.0/)
- [numpy 1.23.5](https://pypi.org/project/numpy/)
- [tensorflow-gpu 2.5.0](https://pypi.org/project/tensorflow-gpu/)
- [keras 2.9.0](https://pypi.org/project/keras/)
- [scipy 1.11.4](https://pypi.org/project/scipy/)
- [spektral 0.6.1](https://pypi.org/project/spektral/)

- [R>=4.1.0](https://www.r-project.org/)
- [reticulate 1.26](https://cran.r-project.org/web/packages/reticulate)
- [doParallel 1.0.17](https://cran.r-project.org/web/packages/doParallel)
- [foreach 1.5.2](https://cran.r-project.org/web/packages/foreach)
- [future 1.33.0](https://cran.r-project.org/web/packages/future)
- [Icens 1.72.0](https://cran.r-project.org/web/packages/Icens)
- [bayesSurv 3.4 ](https://cran.r-project.org/web/packages/bayesSurv)
- [creditmodel 1.3.1](https://cran.r-project.org/web/packages/creditmodel)
- [MASS 7.3-57](https://cran.r-project.org/web/packages/MASS)


## Data

The data used in this research are collected from R package bayesSurv and ADNIMERGE.


## Functions

The program is divided into three sections saved in this repository.

1) graph_function.py: a python code containing fundamental functions for graph processing.

2) train.py: a python code designed to train the GTsurvival model.

3) GTsurv.py: a python code detailing the structure of the GTsurvival model.

3) Main.R: The R code is used to reproduce the prediction results.


## Instructions for Use

The code provided can be used to reproduce the results of GTsurvival.

First, we need to install and load the following R and Python packages.

Second, to reproduce the results, put the files provided in the
working directory and execute the following commands in R:source_python("train.py")

Third, run: Main.R


