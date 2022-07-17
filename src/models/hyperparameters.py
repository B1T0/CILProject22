"""
Here we can store the hyperparameter configurations for each model
"""

from config import config

params = {

    'Contrastive_Pretraining': {
        'lr': 1e-4,
        'batch_size': 4,
        'epochs': 200,
        #backbone
        'embedding_dim': 50,
        'file_path': 'data/raw',
        #model
        'device': None
    },

    'AutoEncoder': {
        'lr': 1e-4, 
        'batch_size': 128,
        'epochs': 50,
        'input_width': 11000,
        'hidden_dims': [128, 32],
        'file_path': 'data/raw/data_train.csv',
    }

}
