import pytorch_lightning as pl 
import torch.nn.functional as F
import torch
import torchmetrics
from torch.optim import Adam
from models.cris_model import CrisModel

class Reweighter(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.net = CrisModel()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
    
    def forward(self, x):
        return self.net(x)

    def format_labels(self, labels):
        labels = torch.unsqueeze(labels, dim=1)
        return torch.tensor(labels, dtype=torch.float)

    def training_step(self, batch, batch_idx):
        predictions = self.forward(batch['features'])
        labels = self.format_labels(batch['label'])
        # Compute Loss
        loss = F.binary_cross_entropy_with_logits(predictions, labels)
        self.log('training_loss', loss)
        # Compute accuracy
        self.train_accuracy(predictions, torch.tensor(labels, dtype=torch.int32))
        self.log('train_acc', self.train_accuracy)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        predictions = self.forward(batch['features'])
        labels = self.format_labels(batch['label'])
        # Compute Loss
        loss = F.binary_cross_entropy_with_logits(predictions, labels)
        self.log('validation_loss', loss)
        # Compute accuracy
        self.val_accracy(predictions, torch.tensor(labels, dtype=torch.int32))
        self.log('validation_acc', self.val_accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        predictions = self.forward(batch['features'])
        labels = self.format_labels(batch['label'])
        # Compute Loss
        loss = F.binary_cross_entropy_with_logits(predictions, labels)
        self.log('test_loss', loss)
        # Compute accuracy
        self.test_accuracy(predictions, torch.tensor(labels, dtype=torch.int32))
        self.log('test_accuracy', self.test_accuracy, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)