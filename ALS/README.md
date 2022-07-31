## Alternating Least Squares
We have two implementions of the ALS-algorithm: [ALS.ipynb](https://github.com/B1T0/CILProject22/blob/main/notebooks/ALS.ipynb) using library calls and [ALS_own.ipynb](https://github.com/B1T0/CILProject22/blob/main/notebooks/ALS_own.ipynb) in which we implemented the ALS algorithm from scratch using the Least Squares formula from the lecture notes. We used the former for our final prediction as the library implementation is more efficient and yields better performance for hyperparameter tuning during grid-search. To run the library-code you need to install the pyspark library (pip install pyspark).
We train the model on the full dataset and build an ensemble with 5 different initializations. In the end we average our predictions.

