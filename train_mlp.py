import json
import os
from random import sample
from pathlib import Path
import sys
import time

from torchsummary import summary
import torch 
import torch.optim as optim
from torch import nn 
import torch 

from config import config
from src.models.hyperparameters import params
from src.data.dataloader_factory import get_classifier_dataloaders
from src.utils.logger import Logger
from src.models.model_factory import get_model
from src.utils.train_utils import * 
import wandb 


def main():
    """
    Main function for training the model.
    """
    print(f"Running in {config['mode']} mode")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = (
        #Path("reports")
        #/ "logs"
        Path(config['encoder_dir'])
        / f"{run_id}_{config['dataset']}_MLP"
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(log_dir)
    print(f"Logging to {log_dir}")

    # get dataloaders
    (
        train_dataloader,
        val_dataloader,
        test_dataloader
    ) = get_classifier_dataloaders(config['encoder_dir'])

    # Create model based on config.py and hyperparameters.py settings
    model = get_model(config["classifier_model"])(**params[config["classifier_model"]])
    # Model summary
    summary(model, (1, params[config["classifier_model"]]["input_size"]))
    
    # Log hyperparameters and config file
    with open(f"{log_dir}/hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(params[config["classifier_model"]], f, ensure_ascii=False, indent=4)
    with open(f"{log_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # Create wandb logger
    wandb.init(project=f"medical_images_latent_space_clf",
        entity="eth-dl-research",
        id=f"{run_id}_{config['classifier_model']}"
    )
    wandb.watch(model)
    wandb.config.update(params[config["classifier_model"]])
    wandb.config.update({'encoder': config['encoder_dir']})

    # Create optimizer and loss function
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(config['class_weights']).cuda())
    optimizer = optim.AdamW(model.parameters(), lr=config["classifier_lr"])
    stopper = EarlyStopper(model=model, log_dir=log_dir, patience=config["classifier_patience"])
    model.cuda()

    # Train model and validate
    for epoch in range(config['classifier_epochs']):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}")
        print(f"===== Training =====")
        # train one epoch 
        train_loss, accuracy, f1_score = train_loop(model, train_dataloader, criterion, optimizer)
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": accuracy,
            # "train_sample_accuracy": sample_accuracy,
            "train_f1_score": f1_score
        })
        # validate model 
        print(f"===== Validation =====")
        val_loss, accuracy, f1_score = validation_loop(model, val_dataloader, criterion, 'val')
        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            # "val_sample_accuracy": sample_accuracy,
            "val_f1_score": f1_score
        })
        if stopper(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
        print("============================================================")
    print('Finished Training')
    print("============================================================")

    # Test model 
    print('Testing model')
    model.load_state_dict(torch.load(f"{log_dir}/classifier.pth"))
    test_loss, accuracy, f1_score = validation_loop(model, test_dataloader, criterion, 'test')
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        # "test_sample_accuracy": sample_accuracy,
        "test_f1_score": f1_score
    })
    print('Finished Testing')
    print('Done')


if __name__ == "__main__":
    main()
