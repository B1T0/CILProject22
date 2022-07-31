# CILProject22

### Authors: Tobias Scheithauer, Jie-Ming Li, Dustin Klebe, Lukas Wolf 

# Table of Contents  
1. [Introduction](#1-introduction)
2. [Models](#2-models)
- [SVD](#svd)
- [SVD++](#svd-1)
- [ALS (ALS)](#als)
- [Mixture-of-Tastes](#mixture-of-tastes)
- [Bias SGD](#bias-sgd)
- [Neural Collaborative Filtering](#neural-collaborative-filtering-ncf)
- [Autoencoder](#autoencoder)
- [Dual Embedding](#dual-embedding)
- [Graph-Encoder](#graph-encoder)
- [Graph-User-Encoder](#graph-user-encoder)
- [Graph-User-Encoder (Attention)](#graph-user-encoder-attention)
3. [Results](#3-results)



# 1. Introduction
In most recommender systems, the number of rated user items is exceeded by the number of total items by several magnitudes. Therefore, the nature of the problem is hard due to it’s inherent sparsity. Today’s recommendation systems often rely on collaborative filtering approaches to compute recommendations for users on a personalised basis. Initial advances in collaborative filtering have been achieved using various blends of matrix factorization approaches, where a rating interaction is represented using latent user and movie variables. Recently, graph-convolutional methods based on spectral i.e. matrix graph representation have achieved SOTA results in the recommendation setting. Despite their seeming differences, we note that many of the classical matrix factorization can be described in the framework of graph convolution. In this project we explore possible augmentations of classical techniques using various graph convolutional techniques. 

The models were implemented for the Computational Intelligence Lab 2022 @ETH Zürich in the Collaborative Filterig [kaggle](https://www.kaggle.com/competitions/cil-collaborative-filtering-2022/overview) competition.

# 2. Models 

### SVD 
For each model we have a single folder. By running the run.py script the pipeline of creating, training and predicting runs through and stores the predictions in CSV-files

### SVD++

### ALS
We have two implementions of the ALS-algorithm: [ALS.ipynb](https://github.com/B1T0/CILProject22/blob/main/ALS/ALS.ipynb) using library calls and [ALS_own.ipynb](https://github.com/B1T0/CILProject22/blob/main/ALS/ALS_own.ipynb) in which we implemented the ALS algorithm from scratch using the Least Squares formula from the lecture notes. In our final implementation in [run.py](https://github.com/B1T0/CILProject22/blob/main/ALS/run.py) we used the former for our submission as the library implementation is more efficient and yields better performance for hyperparameter tuning during grid-search. To run the library-code you need to install the pyspark library (pip install pyspark).
We train the model on the full dataset and build an ensemble with 5 different initializations. In the end we average our predictions.

### Mixture-of-Tastes


### Bias SGD
In this approach we model the significant importance of user- and item-specific ramifications. In order to model these biases, each user u and item i is associated with bias terms. Hyperparameters can be found in the [bias_sgd.py file](https://github.com/B1T0/CILProject22/blob/main/Bias%20SGD/bias_sgd.py). To start training and create submission files, run: 
```
python bias_sgd.py
```


### Neural Collaborative Filtering (NCF)
By running the script [run.py](https://github.com/B1T0/CILProject22/blob/main/NFC/run.py) the model will be built as specified in model.py, and trained on the 5-fold split. In the prediction phase it saves 5 predictions.csv files in the [data-folder](https://github.com/B1T0/CILProject22/tree/main/data), one for each trained model. Afterwards the predictions will be averaged into one ensemble prediciton. The hyperparameters are the ones we used in our final predictions and are the one we found by doing hyperparameter tuning.


### Autoencoder 
In the Autoencoder [directory](https://github.com/B1T0/CILProject22/tree/main/Autoencoder) you can find the implementation of a fully-connected Autoencoder architecture to both encode and decode both user- and item vectors. To run the training procedure and create the submission files, run 
```
python autoencoder.py
```
The hyperparameters can be set in the autoencoders.py file.s

### Dual Embedding 
In the Dual Embedding model, we extend to Neural Collaborative Filtering approach with pretrained user- and item embeddings. In contrast to NCF, we do not learn a look-up table with embeddings, but learn these embeddings in a previous step with distinct learning objective. In the Dual Embedding [directory](https://github.com/B1T0/CILProject22/tree/main/Dual%20Embedding). Hyperparameter configurations can be found in the dual_embedding.py file. Note that you have to specify paths to two pretrained user- and item encoders, respectively. After specifying the paths, to start training and create submission files, run:
```
python dual_embedding.py
```



### Graph-Encoder



### Graph-User-Encoder



### Graph-User-Encoder (Attention)










# 3. Results
| Model                          | RMSE (Training: 5-fold CV) | RMSE (Submission) |
|--------------------------------|----------------------------|-------------------|
| Kaggle Baseline (SVD + ALS)    | -                          | 0.98777           |
| SVD                            | X                          | X                 |
| SVD++^*                        | 0.99712                    | 0.99601           |
| ALS                            | 0.98771                    | 0.98906           |
| Mixture-of-Tastes              | 0.9896                     | 0.98353           |
| Bias SGD                       | 0.98324                    | 0.97794           |
| NCF                            | 0.98641                    | 0.98146           |
| Autoencoder (item)             | 0.99586                    | 0.98381           |
| Autoencoder (user)             | 0.96266                    | 0.97682           |
| Dual Embedding                 | X                          | X                 |
| Graph-Encoder                  | 0.98130                    | 0.98273           |
| Graph-User-Encoder             | 0.99003                    | 0.98626           |
| Graph-User-Encoder (Attention) | 1.00627                    | 1.00889           |
| Ensemble                       | -                          | X                 |
