import numpy as np
import pandas as pd
from subprocess import call
from sklearn.metrics import mean_squared_error
import math
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)


# Load the data
df_full = pd.read_csv('../data_raw/data_train.csv')
df_train = pd.read_csv('../data_raw/cross_validation/train_split_4.csv')
df_test = pd.read_csv('../data_raw/cross_validation/test_split_4.csv')

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

print('Data loading')
full_data = pd.DataFrame(dic_full)
train_data = pd.DataFrame(dic_train)
test_data = pd.DataFrame(dic_test)

train_lib = list(zip(train_data.user_id, train_data.item_id, train_data.rating))
test_lib = list(zip(test_data.user_id, test_data.item_id, test_data.rating))
full_lib = train_lib + test_lib

df_full = spark.createDataFrame(full_lib, ["user", "item", "rating"])
df_train = spark.createDataFrame(train_lib, ["user", "item", "rating"])
df_test = spark.createDataFrame(test_lib, ["user", "item", "rating"])



from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# train and tuning the model
def tune_ALS(train_data, validation_data, maxIter, regParams, ranks):
    """
    Doing Grid Search for tuning the parameters of ALS while keeping track
    of the best parameters, the best model and the metrics
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


# Training
print('Training the model...')
model = tune_ALS(df_train, df_test, maxIter = 15, regParams = [0.05, 0.1, 0.2], ranks = [5, 10, 15, 20, 25, 30])
print('Training complete')
pred = model.transform(df_test)

evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
rmse = evaluator.evaluate(pred)
print('RSME: {}' .format(rmse))


# Submission
# predict ratings
sample_sub = pd.read_csv("../data_raw/sampleSubmission.csv")
to_predict = []
for i, cell_id in enumerate(sample_sub.Id):
    row, col = cell_id.split("_")
    to_predict.append((int(row[1:]), int(col[1:])))
df_sub = spark.createDataFrame(to_predict, ["user", "item"])

ensemble = 5
average = np.asarray([0.0] * len(to_predict))
for i in range(ensemble):
    als = ALS(seed=i).setMaxIter(15).setRank(30).setRegParam(0.1)
    print(f'Training model {i+1}/{ensemble}')
    model = als.fit(df_full)
    predictions = model.transform(df_sub)
    pandas_df = predictions.toPandas()
    pandas_df = pandas_df.sort_values(by=['item', 'user'], ascending=True)
    average += np.asarray(pandas_df['prediction'].values.tolist())
    print(pandas_df[:3])
average = average / ensemble

print(pandas_df[:5])
print(predictions.show())

# write to csv
sample_sub.Prediction = average
sample_sub.to_csv("../data/als.csv", index=False)
sample_sub