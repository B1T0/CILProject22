"""
Return a model class based on the given string 
"""

from src.models.vae import VAE
from src.models.Classifier import MLP


def get_model(model_name: str = "BetaVAE"):
    if model_name == "VAE":
        return VAE
    elif model_name == "BetaVAE":
        return VAE
    elif model_name == "BetaCVAE":
        return VAE
    elif model_name == "MLP":
        return MLP
    else:
        raise Exception("Choose valid model in config.py")
