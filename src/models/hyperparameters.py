"""
Here we can store the hyperparameter configurations for each model
"""

from config import config

params = {

    'GraphAttention': {
        'batch_size': 8,
        'epochs': 300,
        'step_size': 70,
        'gamma' : 0.1,
        'save_period': 50,
        'patience': 30,
        'model_params':{
            'lr': 1e-4,
            'loss' : 'MSE',
            #model_hyper_params
            'mode': 'user_mode', #'movie_mode'
            'latent_dim': 64, # 128
            'graph_hidden': 32,
            'hidden' : 32,
            'n_users': 10000,
                'n_items' : 1000,
            'alpha': 0.2,
            'dropout': 0.2
        }
    },

    'GraphAutoencoder': {

            'batch_size': 8,
            'epochs': 300,
            'step_period': 70,
            'step_size': 70,
            'gamma' : 0.1,
            'save_period': 50,
            'patience': 30,
            'model_params':{
                'lr': 1e-4,
                'loss' : 'MSE',
                #model_hyper_params
                'latent_dim': 48, # 128
                'n_users': 10000,
                'n_items' : 1000
            }
        },
    'GraphUserEncoder': {
            'batch_size': 32,
            'epochs': 200,
            'step_size': 70,
            'gamma' : 0.1,
            'step_period': 70,
            'save_period': 50,
            'patience': 30,
            #model_hyper_params
            'model_params':{
                'lr': 1e-3,
                'loss' : 'MSE',
                #model_hyper_params
                'latent_dim': 16, # 128
                'n_users': 10000,
                'n_items' : 1000,
                'mode' : 'user_mode'
            }
        },

    'SVDGraphAttention': {
        'batch_size': 8,
        'epochs': 300,
        'step_size': 70,
        'gamma' : 0.1,
        'save_period': 50,
        'patience': 30,
        'model_params':{
            'lr': 1e-4,
            'loss' : 'MSE',
            #model_hyper_params
            'n_users': 10000,
            'n_items' : 1000,
            'latent_dim': 64, # 128
            'alpha': 0.2,
            'dropout': 0.2,
            'global_mean': 3.8572805008190647
        }
    },
}
