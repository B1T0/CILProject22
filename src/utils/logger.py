import sys
import torch
from config import config
from src.models.hyperparameters import params
import os
import json


class Logger:
    def __init__(self, print_fp=None):
        self.terminal = sys.stdout
        self.log_file = "out.txt" if print_fp is None else print_fp
        self.encoding = sys.stdout.encoding

        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
            print(f'removed {self.log_file}')

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass


class WandbImageLogger:

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
        for i, batch in enumerate(self.dataloader):
            if i == 5: break
            x, _ = batch
            x = x.view(x.size(0), -1)
            inputs = x.to(self.device)
            outputs = self.model(inputs)
            print(f"inputs shape {inputs.shape}, outputs shape {outputs.shape}")
            # reshape tensor
            outputs = outputs.view(inputs.shape)

            input_imges.append(inputs.cpu().detach().numpy()[0])
            output_imges.append(outputs.cpu().detach().numpy()[0])

        # log to wandb
        for i, img in enumerate(input_imges):
            self.wandb_logger.log_image(
                key="test set",
                images=[input_imges[i], output_imges[i]],
                caption=["input", "reconstructed"]
            )


def log_params(log_dir):
    #with open(f'{log_dir}/hyperparameters.json', 'w', encoding='utf-8') as f:
    #    json.dump(params[config['model']], f, ensure_ascii=False, indent=4)
    #with open(f'{log_dir}/config.json', 'w', encoding='utf-8') as f:
    #    json.dump(config, f, ensure_ascii=False, indent=4)
    pass 


