import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import call
from sklearn.metrics import mean_squared_error
import math
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
conf = SparkConf()
conf.set('spark.executor.memory', '8g')
conf.set('spark.driver.memory', '4g')
#spark = SparkSession(sc).builder

spark = SparkSession(sc).builder.config(conf=conf).getOrCreate()
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


# Load the data
print('Loading data...')
<<<<<<< HEAD
df_full = pd.read_csv('/home/jj/CILProject22/data/raw/data_train.csv')
df_train = pd.read_csv('/home/jj/CILProject22/data/raw/cross_validation/train_split_4.csv')
df_test = pd.read_csv('/home/jj/CILProject22/data/raw/cross_validation/test_split_4.csv')
=======
df_full = pd.read_csv('/home/dustin/Documents/Study/Master2/CILProject22/data_raw/data_train.csv')
df_train = pd.read_csv('/home/dustin/Documents/Study/Master2/CILProject22/data_raw/cross_validation/train_split_4.csv')
df_test = pd.read_csv('/home/dustin/Documents/Study/Master2/CILProject22/data_raw/cross_validation/test_split_4.csv')
>>>>>>> 7b27b5019a91af3748b8a80db2e08f80ab92cd5c

dic_full = {
    'user_id': [int(str(x).partition("_")[0][1:]) for x in df_full['Id']],
    'item_id': [int(str(x).partition("_")[2][1:]) for x in df_full['Id']],
    #'combined': [(str(x).partition("_")[0][1:],str(x).partition("_")[2][1:]) for x in df['Id']],
    'rating': [float(x) for x in df_full['Prediction']],
}
dic_train = {
    'user_id': [int(str(x).partition("_")[0][1:]) for x in df_train['Id']],
    'item_id': [int(str(x).partition("_")[2][1:]) for x in df_train['Id']],
    'rating': [float(x) for x in df_train['Prediction']],
}
dic_test = {
    'user_id': [int(str(x).partition("_")[0][1:]) for x in df_test['Id']],
    'item_id': [int(str(x).partition("_")[2][1:]) for x in df_test['Id']],
    'rating': [float(x) for x in df_test['Prediction']],
}

full_data = pd.DataFrame(dic_full)
train_data = pd.DataFrame(dic_train)
test_data = pd.DataFrame(dic_test)
full_data[:100]




def tune_ALS(train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    train_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    validation_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    maxIter: int, max number of learning iterations
    
    regParams: list of float, one dimension of hyper-param tuning grid
    
    ranks: list of float, one dimension of hyper-param tuning grid
    
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = ALS().setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            # train ALS model
            print(f'Training model with rank {rank} and regularization {reg}')
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and '
          'regularization = {} with RSME = {}'.format(best_rank, best_regularization, min_error))
    return best_model


def make_recommendations(self, fav_movie, n_recommendations):
    """
    make top n movie recommendations
    Parameters
    ----------
    fav_movie: str, name of user input movie
    n_recommendations: int, top n recommendations
    """
    # get data
    movie_user_mat_sparse, hashmap = self._prep_data()
    # get recommendations
    raw_recommends = self._inference(
        self.model, movie_user_mat_sparse, hashmap,
        fav_movie, n_recommendations)
    # print results
    reverse_hashmap = {v: k for k, v in hashmap.items()}
    print('Recommendations for {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance '
              'of {2}'.format(i+1, reverse_hashmap[idx], dist))


print('Preparing data...')
train_lib = list(zip(train_data.user_id, train_data.item_id, train_data.rating))
test_lib = list(zip(test_data.user_id, test_data.item_id, test_data.rating))
full_lib = train_lib + test_lib
print(train_lib[:10])
print(test_lib[:10])

# df_train = spark.createDataFrame(train_lib, ["user", "item", "rating"])
# df_test = spark.createDataFrame(test_lib, ["user", "item", "rating"])
df_full = spark.createDataFrame(full_lib, ["user", "item", "rating"])



# train ALS model
print('Training model...')
als = ALS().setMaxIter(30).setRank(30).setRegParam(0.1)
print(f'Training model with rank {30} and regularization {0.1}')
model = als.fit(df_full)

# predict ratings
print('Predicting ratings...')
sample_sub = pd.read_csv("/home/jj/CILProject22/data/raw/sampleSubmission.csv")
to_predict = []
for cell_id in sample_sub.Id:
    row, col = cell_id.split("_")
    to_predict.append((int(row[1:]), int(col[1:]), 0.0))
df_to_pred = spark.createDataFrame(to_predict, ["user", "item", "rating"])
predictions = model.transform(df_to_pred)
print(predictions.show())
prediction = [pred[0] for pred in list(predictions.select('prediction').collect())]

# write to csv
print('Writing to csv...')
sample_sub.Prediction = prediction
sample_sub.to_csv("../data/als.csv", index=False)
