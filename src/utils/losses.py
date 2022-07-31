import torch 
from torch.autograd import Variable
from torch import nn 

def MaskedMSELoss(inputs, targets, mask=None):
  if mask == None:
    mask = targets != 0
  num_ratings = torch.sum(mask.float())
  criterion = nn.MSELoss(reduction='sum')
  mse = criterion(inputs * mask.float(), targets * mask.float())
  # print(f"inputs {inputs[:5]}, \ntargets {targets[:5]}, \nmask {mask[:5]}")
  # print(f"mse {mse}, num_ratings {num_ratings}")
  return mse, num_ratings