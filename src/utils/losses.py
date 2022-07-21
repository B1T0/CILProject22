import torch 


class MaskedMSELoss(torch.nn.Module):
    """
    Implements a masked MSE loss function where we compute the loss
    only over non-nan/existing entries (marked by mask)
    """
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        if torch.sum(mask) == 0: 
            return torch.tensor(0.)
        else:
            return torch.sum(((input - target) * mask) ** 2.0)  / torch.sum(mask)
