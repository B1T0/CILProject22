"""
MLP to perform classification on the latent space data 
"""
import pytorch_lightning as pl
from torch import nn 
import torch 


class MLP(pl.LightningModule): 
    def __init__(self, input_size=32, output_size=1, hidden_sizes=[32,16,8], lr=1e-5, loss=nn.MSELoss, **kwargs
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size  
        self.hidden_sizes = hidden_sizes
        self.lr = lr 
        self.loss = loss()
        # variable many hidden layers from input size to output size
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)
            )
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.softmax(self.output_layer(x))
        return x

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, y_hat = self._common_step(batch, batch_idx, "train")
        return {"loss": loss, "y_hat": y_hat}

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, y_hat = self._common_step(batch, batch_idx, "valid")
        return {"val_loss": loss, "y_hat": y_hat}

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, y_hat = self._common_step(batch, batch_idx, "test")
        return {"loss": loss, "y_hat": y_hat}

    def _prepare_batch(self, batch):
        """Prepare batch."""
        x, y = batch['x'], batch['y']
        # print(f"x shape {x.shape}")
        x = x.to(self.device)
        # compute the non-nan mask 
        nan_mask = torch.isnan(x)
        # makenans in x to 0 
        x[nan_mask] = 0
        # return x.view(x.size(0), -1)
        return x, nan_mask

    def _common_step(self, batch, batch_idx, stage: str):
        """Common step."""
        x, nan_mask = self._prepare_batch(batch)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        #print(f"{stage} loss: {loss.item()}, type {type(loss)}, torch type {type(loss.data)}")
        return {"loss": loss, "y_hat": y_hat}


if __name__ == '__main__':
    # test the classifier
    # define the classifier
    classifier = MLP(input_size=64, output_size=100, hidden_sizes=[64, 32, 32])
    # print model summary
    from torchsummary import summary
    summary(classifier, input_size=(1, 64))
    # define the input
    x = torch.randn(1, 64)
    print(f"input shape {x.shape}")
    # run the classifier
    y = classifier(x)
    print(f"output shape {y.shape}")
