"""
Here we can store the hyperparameter configurations for each model
"""

from config import config

params = {

    'Acceptor': {
        'lr': 1e-4,
        'batch_size': 16,
        'epochs': 200,
        #backbone
        'backbone': 'None',
        'timestamps': 512,
        'in_channels': 3,
        'out_channels': 3, #backbone output channel
        'kernel_size' : 32,
        'nb_filters' : 64,
        'use_residual': False,
        'backbone_depth' : 4,
        #transformer
        'hidden_dim' : 32,
        'dropout' : False,
        'nheads': 8,
        'dim_feedforward' : 45,
        'enc_layers' : 5,
        'dec_layers' : 5,
        'pre_norm': False,
        #model
        'device': None,
        'position_embedding': 'sine',
        'num_queries': 1,
        'maxpools': [6, 4, 4, 2],

    },

}
