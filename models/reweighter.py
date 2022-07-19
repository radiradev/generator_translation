import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchmetrics
from torch.optim import Adam
from models.modules import CrisModel, tab_transformer
from torchmetrics.functional import accuracy
from tab_transformer_pytorch import TabTransformer


class Reweighter(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        # self.lr = hparams.lr
        self.save_hyperparameters(hparams)
        self.net = CrisModel()

    def forward(self, x):
        return self.net(x)

    def format_labels(self, labels):
        labels = torch.unsqueeze(labels, dim=1)
        return torch.tensor(labels, dtype=torch.float)

    def _shared_step(self, batch, mode):
        features = torch.tensor(batch['features'], dtype=torch.float)
        predictions = self.forward(features)
        labels = self.format_labels(batch['label'])

        # Compute Loss
        weights = batch['weights'].reshape_as(labels)
        loss = F.binary_cross_entropy_with_logits(predictions, labels, weight=weights)
        self.log(f'{mode}_loss', loss)

        # Compute accuracy
        acc = accuracy(predictions, torch.tensor(labels, dtype=torch.int32))
        self.log(f'{mode}_acc', acc)

        return loss, labels

    def training_step(self, batch, batch_idx):
        loss, labels = self._shared_step(batch, mode='training')
        n_positive_examples = torch.sum(labels == 1.0)
        self.log('positive_examples', n_positive_examples)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels = self._shared_step(batch, mode='validation')
        return loss

    def test_step(self, batch, batch_idx):
        loss, labels = self._shared_step(batch, mode='test')
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), 1e-4)
