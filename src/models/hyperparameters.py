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

}
