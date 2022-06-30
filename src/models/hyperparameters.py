"""
Here we can store the hyperparameter configurations for each model
"""

from config import config

params = {
    "BasicAE": {
        "input_width": config["input_width"],
        "input_height": config["input_height"],
        "hidden_dims": [1024, 512, 340, 70],
        "latent_dim": 70,  # set to last hidden dim for logging
        "lr": config["lr"],
    },
    "VAE": {
        "ae_module": "BasicAEModule",
        "input_width": config["input_width"],
        "input_height": config["input_height"],
        "hidden_dims": [1024, 512, 340, 70],
        "latent_dim": 70,  # set to last hidden dim for logging
        "lr": config["lr"],
    },
    "BetaVAE": {
        "beta": 1.5,
        "ae_module": "BasicAEModule",
        "input_width": config["input_width"],
        "input_height": config["input_height"],
        "latent_dim": 340,
        "lr": config["lr"],
    },
    "BetaCVAE": {
        "beta": 1.0,
        "gamma": 10,
        "ae_module": "BasicAEModule",
        "input_width": config["input_width"],
        "input_height": config["input_height"],
        "latent_dim": 340,
        "lr": config["lr"],
    },
    "MLP": {
        "input_size": 340,
        "output_size": 170,
        "hidden_sizes": [200, 100, 100, 100, 100],
        "lr": config["classifier_lr"]
    }
}
