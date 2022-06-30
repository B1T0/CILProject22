import sys
import os
import torch 
import wandb 
from config import config 
import numpy as np 


class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path + "console.out", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class WandbImageLogger():

    def __init__(self, model, dataloader, wandb_logger, seed=42) -> None:
        self.model = model 
        self.dataloader = dataloader 
        self.wandb_logger = wandb_logger
        self.seed = seed 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def log_images(self):
        input_imges = [] 
        output_imges = []

        # get predictions on the first 5 images 
        self.model.eval().to(self.device)   
        test_batch = next(iter(self.dataloader))
        #x, conditions = test_batch[0:2]
        #print(f"test batch shape {np.array(test_batch).shape}")
        
        for i in range(5):
            inputs, conditions = test_batch[0:2]
            #print(f"inputs shape {inputs.shape}")
            shape = inputs.shape 
            # Forward pass 
            outputs = self.model.predict_step(test_batch, i)
            #print(f"outputs shape: {outputs.shape}")
            # reshape tensors for images 
            outputs = outputs.view(shape)
            # scale back to [0,255] range
            inputs = inputs * 255
            outputs = outputs * 255
            #print(f"inputs shape {inputs.shape}, outputs shape {outputs.shape}")
            input_imges.append(inputs.cpu().detach().numpy()[i])
            output_imges.append(outputs.cpu().detach().numpy()[i])
    

        for i, img in enumerate(input_imges):
            self.wandb_logger.log_image(
                                key="test set", 
                                images=[input_imges[i], output_imges[i]], 
                                caption=["input", "reconstructed"] 
                                )


        

