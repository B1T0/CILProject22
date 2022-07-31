# CILProject22

### Authors: Tobias Scheithauer, Jie-Ming Li, Dustin Klebe, Lukas Wolf 

# Table of Contents  
### 1. Introduction 
### 2. Models
#### [ALS (ALS)](#ALS)
#### [SVD](#SVD)
#### NCF
#### Bias SGD 
#### Autoencoder
#### Dual Embedding
### 3. Results



# 1. Introduction
In most recommender systems, the number of rated user items is exceeded by the number of total items by several magnitudes. Therefore, the nature of the problem is hard due to it’s inherent sparsity. Today’s recommendation systems often rely on collaborative filtering approaches to compute recommendations for users on a personalised basis. Initial advances in collaborative filtering have been achieved using various blends of matrix factorization approaches, where a rating interaction is represented using latent user and movie variables. Recently, graph-convolutional methods based on spectral i.e. matrix graph representation have achieved SOTA results in the recommendation setting. Despite their seeming differences, we note that many of the classical matrix factorization can be described in the framework of graph convolution. In this project we explore possible augmentations of classical techniques using various graph convolutional techniques. 

The models were implemented for the Computational Intelligence Lab 2022 @ETH Zürich at the Collaborative Filterig [kaggle](https://www.kaggle.com/competitions/cil-collaborative-filtering-2022/overview) competition.

# 2. Models 
## ALS
We have two implementions of the ALS-algorithm: [ALS.ipynb](https://github.com/B1T0/CILProject22/blob/main/ALS/ALS.ipynb) using library calls and [ALS_own.ipynb](https://github.com/B1T0/CILProject22/blob/main/ALS/ALS_own.ipynb) in which we implemented the ALS algorithm from scratch using the Least Squares formula from the lecture notes. In our final implementation in [run.py](https://github.com/B1T0/CILProject22/blob/main/ALS/run.py) we used the former for our submission as the library implementation is more efficient and yields better performance for hyperparameter tuning during grid-search. To run the library-code you need to install the pyspark library (pip install pyspark).
We train the model on the full dataset and build an ensemble with 5 different initializations. In the end we average our predictions.

## Neural Collaborative Filtering (NCF)
By running the script run.py the model will be built as specified in model.py, and trained on the 5-fold split. In the prediction phase it saves 5 predictions.csv files in the [data-folder](https://github.com/B1T0/CILProject22/tree/main/data), one for each trained model. Afterwards the predictions will be averaged into one ensemble prediciton. The hyperparameters are the ones we used in our final predictions and are the one we found by doing hyperparameter tuning.

## SVD 
For each model we have a single folder. By running the run.py script the pipeline of creating, training and predicting runs through and stores the predictions in CSV-files

## Bias SGD

## Autoencoder 

## Dual Embedding 



# 3. Results
