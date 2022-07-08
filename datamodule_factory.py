"""
Create dataloaders depending on settings in config.py
"""

from config import config
from src.models.hyperparameters import params
from src.data.FIRST.first_datamodule import FIRST_DataModule
from src.data.Contrastive.graph_datamodule import Graph_DataModule
from pathlib import Path


def get_datamodule():
    if config['dataset'] == 'contrastive':
        return Graph_DataModule(
            params[config['model']]['file_path'],
            batch_size=params[config['model']]['batch_size']
            )
    else:
        raise NotImplementedError("Choose valid dataset in config.py")


if __name__ == '__main__':
    print(f"Loading data")
    datamodule = get_datamodule()
    datamodule.prepare_data()
    datamodule.setup()
    # test dataloader
    for batch in datamodule.train_dataloader():
        print(batch[0].shape)
        print(batch[1].shape)
        break
