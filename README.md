# CILProject22

## Alternating Least Squares
We have two implementions of the ALS-algorithm: [ALS.ipynb](https://github.com/B1T0/CILProject22/blob/main/notebooks/ALS.ipynb) using library calls and [ALS_own.ipynb](https://github.com/B1T0/CILProject22/blob/main/notebooks/ALS_own.ipynb) in which we implemented the ALS algorithm from scratch using the Least Squares formula from the lecture notes. We used the former for our final prediction as the library implementation is more efficient and yields better performance for hyperparameter tuning during grid-search. To run the library-code you need to install the pyspark library (pip install pyspark).
We train the model on the full dataset and build an ensemble with 5 different initializations. In the end we average our predictions.

## Neural Collaborative Filtering
The Code of NFC-models is in the [NCF-folder](https://github.com/B1T0/CILProject22/tree/main/src/models/NCF). Running the script train.py will build the model specified in model.py, train it on the 5-fold split and return 5 predictions in the [data-folder](https://github.com/B1T0/CILProject22/tree/main/data), one for each trained model. The hyperparameters are the ones we used in our final predictions are doing hyperparameter tuning.
To average the predictions into one ensemble prediciton run the script load_model_predict.py.
