import pytorch_lightning as pl
import torch
from util import get_model


class Trainer:

    def __init__(self, model_name: str, model_params: dict):
        self.model_name = model_name
        self.params = model_params
        self.model = get_model(self.model_name, model_params)
        self.trainer = pl.Trainer()

    def train(self):
        self.trainer.train(self.model)

    def test(self):
        self.trainer.test(self.model)

    def predict(self, X: torch.tensor):
        self.trainer.predict(self.model, )
