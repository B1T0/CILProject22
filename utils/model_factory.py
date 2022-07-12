"""
Return a model class based on the given string
"""

from src.models.DETRtime.model import build


def get_model(args, model_name: str = 'DETRtime'):
    if model_name == 'DETRtime':
        model, _  = build(args)
        return model
    else:
        raise Exception("Choose valid model in config.py")
