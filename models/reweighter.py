import pytorch_lightning as pl 
import torch.nn.functional as F
import torch

from torch.optim import Adam
from models.cris_model import CrisModel

class Reweighter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = CrisModel()
    
    def forward(self, x):
        return self.net(x)

    def format_labels(self, labels):
        labels = torch.unsqueeze(labels, dim=1)
        return torch.tensor(labels, dtype=torch.float)

    def training_step(self, batch, batch_idx):
        predictions = self.forward(batch['features'])
        labels = self.format_labels(batch['label'])
        loss = F.binary_cross_entropy_with_logits(predictions, labels)
        self.log('training_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)