# CILProject22

# Table of Contents  
### 1. Introduction 
### 2. Models
#### ALS 
#### SVD 
#### NCF



# Introduction
In most recommender systems, the number of
rated user items is exceeded by the number of total items
by several magnitudes. Therefore, the nature of the problem is
hard due to it’s inherent sparsity. Today’s recommendation
systems often rely on collaborative filtering approaches to
compute recommendations for users on a personalised basis.
Initial advances in collaborative filtering have been achieved
using various blends of matrix factorization approaches, where
a rating interaction is represented using latent user and movie
variables. Recently, graph-convolutional methods based on
spectral i.e. matrix graph representation have achieved SOTA
results in the recommendation setting. Despite their seeming
differences, we note that many of the classical matrix factoriza-
tion can be described in the framework of graph convolution.
In this project we explore possible augmentations of classical
techniques using various graph convolutional techniques

# Models 
## ALS 


## SVD 
For each model we have a single folder. By running the run.py script the pipeline of creating, training and predicting runs through and stores the predictions in CSV-files
