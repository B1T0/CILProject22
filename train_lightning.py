import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from config import config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from src.utils.model_factory import get_model
from src.models.hyperparameters import params
import time
from src.utils.logger import Logger, log_params
import os
from src.data.UserSample.user_datamodule import User_DataModule
import logging
from src.utils.utils import inference_for_submission, write_submission, eval_submission
import torch 


def main():

    predictions = None # predictions for submission
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_{config['model']}"

    for i in range(5): # iterate over the splits 
        print(f"----- Starting split {i}")

        # Create model directory and Logger
        log_dir = f"reports/logs/{run_id}/{config['model']}_split_{i}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
        # Create logging file
        logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
        logging.info("Started logging.")

        # Obtain datamodule based on config settings for dataset
        data_module = User_DataModule(
            split_number=i, 
            data_dir=params[config['model']]['file_path'], 
            batch_size=params[config['model']]['batch_size'], 
            item_based=params[config['model']]['item_based'],
            scale=params[config['model']]['scale']
            )
        logging.info("Created data module.")

        # Create model based on config.py and hyperparameters.py settings
        # changed to include model factory
        model = get_model(params[config['model']], config['model'])

        logging.info("Created model.")

        # Log hyperparameters and config file
        log_params(log_dir)

        # Create logger 
        tb_logger = TensorBoardLogger(log_dir)
        tb_logger.log_hyperparams(params[config['model']])  # log hyperparameters
        
        # Create trainer 
        trainer = pl.Trainer(
            accelerator="gpu",  # cpu or gpu
            devices=-1,  # -1: use all available gpus, for cpu e.g. 4
            enable_progress_bar=True,  # disable progress bar
            precision=16, # 16 bit float precision for training
            #logger=[tb_logger, wandb_logger],  # log to tensorboard and wandb
            logger = [tb_logger],
            max_epochs=params[config['model']]['epochs'],  # max number of epochs
            callbacks=[
                EarlyStopping(monitor="valid_loss", patience=15),  # early stopping
                ModelCheckpoint(log_dir, monitor='valid_loss', save_top_k=1),  # save best model
                TQDMProgressBar(refresh_rate=200),
                LearningRateMonitor(logging_interval='step')
            ],
            auto_lr_find=True  # automatically find learning rate
        )

        print("Start training.")
        trainer.fit(model, data_module)  # train the model
        # save best model 
        #trainer.save_checkpoint(f"{log_dir}/best_model.ckpt")
        print("Finished training.")
        #trainer.test(model, data_module)  # test the model
        print("Finished testing.")
        # Perform inference on the whole dataset and save the prediction as submission file 
        split_predictions = inference_for_submission(
            model, 
            data_module,
            save_path=f"{log_dir}/submission.csv", 
            save_rounded=False,
            remove_bias=False,
        )

        if predictions is None:
            predictions = split_predictions
        else:
            predictions = predictions + split_predictions

        print(f"----- Finished inference for split {i}")

    print(f"Finished all splits")

    # Write submission file
    write_submission(
        predictions / 5, 
        '/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', 
        f"reports/logs/{run_id}/all_split_submission.csv"
        )

    # Evaluate submission file
    print(f"all submission validation score: {eval_submission(predictions / 5, data_module)}")



if __name__ == "__main__":
    main()

