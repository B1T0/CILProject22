import json
import os
import sys
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torchsummary import summary

from config import config
from src.models.hyperparameters import params
from src.data.dataloader_factory import get_autoencoder_dataloaders
from src.utils.logger import Logger, WandbImageLogger
from src.models.model_factory import get_model


def main():
    """
    Main function for training the model.
    """
    print(f"Running in {config['mode']} mode")
    print(f"Using dataset {config['dataset']}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_{config['model']}_ld_{params[config['model']]['latent_dim']}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(f"{log_dir}/")
    print(f"Logging to {log_dir}")

    # Create dataloaders
    (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
    ) = get_autoencoder_dataloaders()

    # Create model based on config.py and hyperparameters.py settings
    model = get_model(config["model"])(**params[config["model"]])
    # Model summary
    summary(model, (1, config["input_width"], config["input_height"]))
    
    # Log hyperparameters and config file
    with open(f"{log_dir}/hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(params[config["model"]], f, ensure_ascii=False, indent=4)
    with open(f"{log_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # Run the model
    tb_logger = TensorBoardLogger(f"{log_dir}/", name=f"{run_id}_{config['model']}")
    tb_logger.log_hyperparams(params[config["model"]])  # log hyperparameters
    wandb_logger = WandbLogger(
        project=f"medical_images",
        entity="eth-dl-research",
        save_dir=f"reports/logs/{run_id}_{config['model']}_ld_{params[config['model']]['latent_dim']}",
        id=f"{run_id}_{config['model']}"
    )
    trainer = pl.Trainer(
        accelerator="cpu",  # cpu or gpu
        devices=4,  # -1: use all available gpus, for cpu e.g. 4
        enable_progress_bar=True,  # disable progress bar
        # progress_bar_refresh_rate=500,  # show progress bar every 500 iterations
        # precision=16,  # 16 bit float precision for training
        logger=[tb_logger, wandb_logger],  # log to tensorboard and wandb
        log_every_n_steps=1,  # log every n steps
        max_epochs=config["epochs"],  # max number of epochs
        callbacks=[
            EarlyStopping(
                monitor="valid_loss",
                patience=config['patience'],
                check_finite=True
                ),
            ModelSummary(max_depth=3),  # model summary
            ModelCheckpoint(log_dir, monitor='valid_loss', save_top_k=1) # save best model
            ],
        auto_lr_find=True,  # automatically find learning rate
        detect_anomaly=True,  # detect anomalies in the training data
        # num_sanity_val_steps=0,  # number of validation steps for sanity check
    )
    print(f"Starting training")
    trainer.fit(model, train_dataloader, validation_dataloader)  # train the model
    print(f"Finished training")
    trainer.test(model, test_dataloader)  # test the model

    print(f"Logging images to wandb")
    WandbImageLogger(model, test_dataloader, wandb_logger).log_images()
    print(f"Finished logging images to wandb")
    print(f"Finished run. Logs can be found in {log_dir}")


if __name__ == "__main__":
    main()
