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
from src.utils.logger import Logger, log_params, WandbImageLogger
import os
from datamodule_factory import get_datamodule
from src.data.UserSample.user_dataloader import get_user_dataloaders
import logging
from src.utils.utils import inference_for_submission, read_data, write_submission
import torch 
from src.utils.losses import MaskedMSELoss
from src.models.torch_autoencoder import AutoEncoder 
import json 



def _prepare_batch(batch):
        """Prepare batch."""
        x = batch[0]
        # print(f"x shape {x.shape}")
        x = x.cuda()
        # compute the non-nan mask 
        nan_mask = torch.isnan(x).cuda()
        # makenans in x to 0 
        x[nan_mask] = 0
        # return x.view(x.size(0), -1)
        return x, nan_mask

def train_epoch(training_loader, optimizer, model, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, mask = _prepare_batch(data[0])

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, inputs, ~mask)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    avg_loss = running_loss / len(training_loader)

    return avg_loss

def test_epoch(test_dataloader, model, loss_fn):
    model.eval() 

    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(test_dataloader):
        # Every data instance is an input + label pair
        inputs, mask = _prepare_batch(data[0])

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, inputs, ~mask)

        # check if loss is nan 
        if torch.isnan(loss).any():
            print(f"nan in loss")

        # Gather data and report
        running_loss += loss.item()

    avg_loss = running_loss / len(test_dataloader)

    model.train()
    return avg_loss


def train_model(model, optimizer, loss_fn, train_dataloader, val_dataloader, epochs, run_id, log_dir, patience=15):
    # Train model
    model = model.cuda()
    min_val_loss = float('inf')
    early_stopping_count = 0 
    for epoch_index in range(epochs):
        print(f"Starting epoch {epoch_index}")
        # Train model
        train_loss = train_epoch(train_dataloader, optimizer, model, loss_fn)
        print(f"Avg train loss {train_loss}")
        # Validate model
        val_loss = test_epoch(val_dataloader, model, loss_fn)
        print(f"Avg val loss {val_loss}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            # save model 
            torch.save(model.state_dict(), f"{log_dir}/best_model.pt")
            print(f"Saved best model")
        else:
            early_stopping_count += 1
            if early_stopping_count >= patience:
                print("Early stopping in epoch {}".format(epoch_index))
                break
        print(f"Finished epoch {epoch_index}")
    print(f"Finished training")
    
    # return best model 
    model.load_state_dict(torch.load(f"{log_dir}/best_model.pt"))
    return model



def main():
    predictions = None # predictions for submission
    run_id = time.strftime("%Y%m%d-%H%M%S")

    for i in range(5): # iterate over the splits 
        print(f"Starting split {i}")

        # Create model directory and Logger
        log_dir = f"reports/logs/{run_id}/{config['model']}_split_{i}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
        # Create logging file
        logging.basicConfig(filename=f"{log_dir}/info.log", encoding='utf-8', level=logging.INFO)
        logging.info("Started logging.")

        # Obtain datamodule based on config settings for dataset
        train_loader, val_loader, inference_tensor, inference_scaler = get_user_dataloaders(params[config['model']]['file_path'], i, params[config['model']]['batch_size'])
        model = AutoEncoder()

        # Log hyperparameters and config file as json 
        with open(f"reports/logs/{run_id}/config.json", "w") as f:
            f.write(json.dumps(config, indent=4))
        with open(f"reports/logs/{run_id}/params.json", "w") as f:
            f.write(json.dumps(params[config['model']], indent=4))


        # Create optimizer 
        optimizer = torch.optim.Adam(model.parameters(), lr=params[config['model']]['lr'], weight_decay=params[config['model']]['weight_decay'])
        loss_fn = MaskedMSELoss()

        # Train model 
        model = train_model(model, optimizer, loss_fn, train_loader, val_loader, epochs=params[config['model']]['epochs'], run_id=run_id, log_dir=log_dir)

        # Inference 
        split_predictions = inference_for_submission(model, data=inference_tensor, scaler=inference_scaler, save_path=f"{log_dir}/submission.csv", save_rounded=True)

        print(f"split predictions ")
        if predictions is None:
            predictions = split_predictions
        else:
            predictions = predictions + split_predictions

        print("Finished inference.")
        print(f"Finished split {i}")
    print(f"Finished all splits")

    # Write submission file
    write_submission(
        predictions / 5, 
        '/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', 
        f"reports/logs/{run_id}/submission.csv"
        )




if __name__ == "__main__":
    main()
