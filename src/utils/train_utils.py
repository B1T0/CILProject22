import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torchsummary import summary
import torch 
import torch.optim as optim
from torch import nn 
import torch 
from sklearn.metrics import f1_score, classification_report
import numpy as np 

import os 


def compute_label_accuracy(predictions, labels):
    """
    Computes the label accuracy, i.e. whether we classify 0/1 correctly 
    on the all labels concatenated (not per sample)
    """
    # round to nearest integer
    prediction_flat = torch.flatten(predictions.round())
    labels_flat = torch.flatten(labels)
    # compute how many are correct after rounding to nearest integer
    correct = prediction_flat.eq(labels_flat).sum().item()
    return correct / len(labels_flat)

def compute_sample_accuracy(predictions, labels):
    """
    Computes the sample accuracy, i.e. whether we classify a whole sample correctly 
    """
    # round to nearest integer
    predictions = predictions.round()
    # check how many predictions are correct
    correct = torch.eq(predictions, labels)
    correct = list(correct.cpu().detach().numpy())
    # check how many predictions are correct for all labels within one sample 
    count_correct = sum([np.all(c) for c in correct])
    return count_correct / len(correct)

def compute_f1(predictions, labels):
    """
    Computes the f1 score on concatenated labels and predictions, i.e. on the 
    classes 0/1 (not per sample)
    """
    # round to nearest integer
    prediction_flat = torch.flatten(predictions.round())
    labels_flat = torch.flatten(labels)
    # convert tensor to numpy 
    prediction_flat = prediction_flat.cpu().detach().numpy()
    labels_flat = labels_flat.cpu().detach().numpy()
    f1 = f1_score(prediction_flat, labels_flat, average='macro')
    return f1 

def print_classification_report(predictions, labels, mode):
    """
    Prints the sklearn classification report on all labels concatenated, i.e. on the 
    classes 0/1 (not per sample)
    """
    # round to nearest integer
    prediction_flat = torch.flatten(predictions.round())
    labels_flat = torch.flatten(labels)
    # convert tensor to numpy 
    prediction_flat = prediction_flat.cpu().detach().numpy()
    labels_flat = labels_flat.cpu().detach().numpy()
    print(f'{mode} classification report: \n{classification_report(prediction_flat, labels_flat)}')

def train_loop(model, train_dataloader, criterion, optimizer):
    # Train model
    running_loss = 0.0
    all_predictions = torch.empty(0, dtype=torch.float).cuda()
    all_labels = torch.empty(0, dtype=torch.float).cuda()
    for i, data in enumerate(train_dataloader, 0):
        # inference and optimization 
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        # collect metrics 
        all_predictions = torch.cat((all_predictions, outputs), 0)
        all_labels = torch.cat((all_labels, labels), 0)
        running_loss += loss.item()

    # compute metrics and print them
    train_loss = running_loss / len(train_dataloader)
    accuracy = compute_label_accuracy(all_predictions, all_labels)
    f1_score = compute_f1(all_predictions, all_labels)
    sample_accuracy = compute_sample_accuracy(all_predictions, all_labels)
    print(f'train loss: {train_loss:.3f}')
    print(f'train accuracy: {accuracy:.3f}')
    print(f'train f1 score: {f1_score:.3f}')
    print(f'train sample accuracy: {sample_accuracy:.3f}')
    #print_classification_report(all_predictions, all_labels, 'train')
    return train_loss, accuracy, sample_accuracy, f1_score

def validation_loop(model, val_dataloader, criterion, mode):
    # Validate model
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        all_predictions = torch.empty(0, dtype=torch.float).cuda()
        all_labels = torch.empty(0, dtype=torch.float).cuda()
        for i, data in enumerate(val_dataloader):
            # inference 
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            # metrics 
            running_loss += loss.item()
            all_predictions = torch.cat((all_predictions, outputs), 0)
            all_labels = torch.cat((all_labels, labels), 0)
        
        # compute metrics and print them
        val_loss = running_loss / len(val_dataloader)
        accuracy = compute_label_accuracy(all_predictions, all_labels)
        f1_score = compute_f1(all_predictions, all_labels)
        sample_accuracy = compute_sample_accuracy(all_predictions, all_labels)
        print(f'{mode} loss: {val_loss:.3f}')
        print(f'{mode} accuracy: {accuracy:.3f}')
        print(f'{mode} f1 score: {f1_score:.3f}')
        print(f'{mode} sample accuracy: {sample_accuracy:.3f}')
        if mode == 'test':
            print_classification_report(all_predictions, all_labels, mode)

    model.train()
    return val_loss, accuracy, sample_accuracy, f1_score


class EarlyStopper:
    def __init__(self, model, log_dir, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.model = model  
        self.log_dir = log_dir

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss >= self.best_score:
            self.counter += 1
            print(f'EarlyStopper: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  
        else:   
            print(f"Validation loss improved from {self.best_score:.3f} to {val_loss:.3f}. Saving model.")
            self.best_score = val_loss
            self.counter = 0
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model.pth'))
        return self.early_stop

            