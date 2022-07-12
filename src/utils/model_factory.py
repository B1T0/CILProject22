"""
Return a model class based on the given string
"""
from src.models.CSE.model import model

def get_model(args, model_name: str):
    if model_name == 'Contrastive_Pretraining':
        return model(args['embedding_dim'])
    else:
        raise Exception("Choose valid model in config.py")
