## Alternating Least Squares
The Implementation of the ALS-algorithm can be found in the [ALS-folder](https://github.com/B1T0/CILProject22/tree/main/ALS).
We have two implementions of the ALS-algorithm: [ALS.ipynb](https://github.com/B1T0/CILProject22/blob/main/ALS/ALS.ipynb) using library calls and [ALS_own.ipynb](https://github.com/B1T0/CILProject22/blob/main/ALS/ALS_own.ipynb) in which we implemented the ALS algorithm from scratch using the Least Squares formula from the lecture notes. In our final implementation in [run.py](https://github.com/B1T0/CILProject22/blob/main/ALS/run.py) we used the former for our submission as the library implementation is more efficient and yields better performance for hyperparameter tuning during grid-search. To run the library-code you need to install the pyspark library (pip install pyspark).
We train the model on the full dataset and build an ensemble with 5 different initializations. In the end we average our predictions.

