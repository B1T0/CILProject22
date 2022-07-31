## Neural Collaborative Filtering
The Code of NFC-models is in the [NCF-folder](https://github.com/B1T0/CILProject22/tree/main/src/models/NCF). Running the script train.py will build the model specified in model.py, train it on the 5-fold split and return 5 predictions in the [data-folder](https://github.com/B1T0/CILProject22/tree/main/data), one for each trained model. The hyperparameters are the ones we used in our final predictions are doing hyperparameter tuning.
To average the predictions into one ensemble prediciton run the script load_model_predict.py.
