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

    'ItemAutoEncoder': {
        # model
        'input_width': 1000,
        'hidden_dims': [128, 16],
        'activation': 'relu',
        # training
        'lr': 1e-4, 
        'batch_size': 16,
        'epochs': 50,
        'weight_decay': 1e-4,
        'dropout': 0.25,
        'patience': 5,
        # dat
        'file_path': 'data/raw',
        'scale': False,
        'item_based': True,
    },

    'UserAutoEncoder': {
        # model
        'input_width': 1000,
        'hidden_dims': [128, 16],
        'activation': 'relu',
        # training
        'lr': 1e-3, 
        'batch_size': 32,
        'epochs': 50,
        'weight_decay': 1e-3,
        'dropout': 0.5,
        'patience': 5,
        # dat
        'file_path': 'data/raw',
        'scale': False,
        'item_based': False,  
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
