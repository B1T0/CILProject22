import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from config import config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.utils.model_factory import get_model
from src.models.hyperparameters import params
import time
from src.utils.logger import Logger, log_params, WandbImageLogger
import os
from datamodule_factory import get_datamodule
import logging


def main():
    # Create model directory and Logger
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_{config['model']}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", encoding='utf-8', level=logging.INFO)
    logging.info("Started logging.")

    # Obtain datamodule based on config settings for dataset
    data_module = get_datamodule()
    logging.info("Created data module.")

    # Create model based on config.py and hyperparameters.py settings
    # changed to include model factory
    model = get_model(params[config['model']], config['model'])
    logging.info("Created model.")
    # print model summary
    # summary(model, (config['input_height'], config['input_width']))

    # Log hyperparameters and config file
    log_params(log_dir)

    # Run the model
    tb_logger = TensorBoardLogger("./reports/logs/",
                                  name=f"{run_id}_{config['model']}"
                                  )
    tb_logger.log_hyperparams(params[config['model']])  # log hyperparameters
    # wandb_logger = WandbLogger(project=f"{config['dataset']}",
    #                            entity="deepseg",
    #                            save_dir=f"reports/logs/{run_id}_{config['model']}",
    #                            id=f"{run_id}_{config['model']}"
    #                            )
    trainer = pl.Trainer(accelerator="gpu",  # cpu or gpu
                         devices=-1,  # -1: use all available gpus, for cpu e.g. 4
                         enable_progress_bar=False,  # disable progress bar
                         # progress_bar_refresh_rate=500, # show progress bar every 500 iterations
                         # precision=16, # 16 bit float precision for training
                         #logger=[tb_logger, wandb_logger],  # log to tensorboard and wandb
                         logger = [tb_logger],

                         max_epochs=params[config['model']]['epochs'],  # max number of epochs
                         callbacks=[EarlyStopping(monitor="Validation Loss", patience=20),  # early stopping
                                    ModelSummary(max_depth=1),  # model summary
                                    ModelCheckpoint(log_dir, monitor='Validation Loss', save_top_k=1)  # save best model
                                    ],
                         auto_lr_find=True  # automatically find learning rate
                         )
    logging.info("Start training.")
    trainer.fit(model, data_module)  # train the model
    logging.info("Finished training.")
    trainer.test(model, data_module)  # test the model

    # Log test images to wandb
    # inference_loader = DataLoader(data_module, batch_size=1, shuffle=False, num_workers=2)
    # WandbImageLogger(model, inference_loader, wandb_logger).log_images()


if __name__ == "__main__":
    main()
