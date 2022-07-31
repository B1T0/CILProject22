"""
Return a model class based on the given string
"""
from src.models.CSE.model import Model
from src.models.lightning_autoencoder import AutoEncoder
from src.models.mlp import MLP

def get_model(args, model_name: str):
    #print(f"Loading model {model_name}")
    if model_name == 'Contrastive_Pretraining':
        return Model(args['embedding_dim'])
    elif model_name == 'UserAutoEncoder' or model_name == 'ItemAutoEncoder':
        return AutoEncoder(**args)
    elif model_name == 'MLP':
        return MLP(**args)
    else:
        raise Exception("Choose valid model in config.py")
