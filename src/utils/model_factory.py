"""
Return a model class based on the given string
"""
from src.models.CSE.model import Model
from src.models.autoencoder import AutoEncoder

def get_model(args, model_name: str):
    if model_name == 'Contrastive_Pretraining':
        return Model(args['embedding_dim'])
    elif model_name == 'AutoEncoder':
        return AutoEncoder(**args)
    else:
        raise Exception("Choose valid model in config.py")
