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
        'lr': 1e-5, 
        'batch_size': 64,
        'epochs': 100,
        'weight_decay': 1e-2,
        'input_width': 1000,
        'hidden_dims': [128, 16],
        'file_path': 'data/raw',
    },

    'MLP': {
        'lr': 1e-4,
        'batch_size': 64,
        'epochs': 50,
        'input_size': 11000,
        'hidden_dims': [512, 128, 32],
        'output_size': 1,
        'file_path': 'data/raw',
    }
}
