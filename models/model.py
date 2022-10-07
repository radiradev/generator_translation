import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch.optim import Adam
from models.modules import ParticleFlowNetwork
from torchmetrics.functional import accuracy, f1_score
from src.utils.funcs import compute_histogram

class LightningModel(pl.LightningModule):

    def __init__(self, learning_rate=None, batch_size=None):
        super().__init__()
        self.batch_size = batch_size
        # self.lr = hparams.lr
        self.learning_rate = learning_rate
        # self.save_hyperparameters(learning_rate)
        self.net = ParticleFlowNetwork(input_dims=4, num_classes=2, transform_to_pt=True)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    def kl_divergence(self, p, q, reduction='sum'):
        log_p = torch.where(p != torch.tensor(0.0), torch.log(p), torch.tensor(0.0))
        return F.kl_div(log_p, q, reduction=reduction)
    
    def forward(self, x):
        return self.net(x)

    def format_labels(self, labels):
        labels = torch.unsqueeze(labels, dim=1)
        return torch.tensor(labels, dtype=torch.long)

    def _shared_step(self, batch, mode):
        features = torch.tensor(batch['features'], dtype=torch.float)
        predictions = self.forward(features)
        labels = self.format_labels(batch['label'])
        
        # We only use input weights for GiBUU
        # weights[labels == 1]/torch.sum(weights[labels == 1])
        # weights = batch['weights'].reshape_as(labels)

        loss = self.loss(predictions, labels.squeeze())

        self.log(f'{mode}_loss', loss)

        # Compute accuracy
        acc = accuracy(predictions, torch.tensor(labels, dtype=torch.int32))
        f1 = f1_score(predictions, torch.tensor(labels, dtype=torch.int32))
        self.log(f'{mode}_acc', acc)
        self.log(f'{mode}_f1_score', f1)

        return loss, labels

    def training_step(self, batch, batch_idx):
        loss, labels = self._shared_step(batch, mode='training')
        n_positive_examples = torch.sum(labels == 1.0)
        self.log('positive_examples', n_positive_examples)
        return loss
    
    def compute_kls(self, dataset, dist_names, weights):
        """Computes KL divergence on a list of histograms
        Only works for unweighted datasets

        Args:
            dataset (torch.Dataset): Validation dataset object 
            dist_names (list[str]): names of the distributions we are computing KL divergence
   
        Returns:
            metric_dict: Dictionary containing the KL divergence between both generators
        """
        val_distributions = dataset.validation_variables
        
        #compute histograms
        val_a = val_distributions[dataset.generator_a]
        val_b = val_distributions[dataset.generator_b]
        
                      
        # Transform to torch tensor and compute KL divergence:
        kl_div = {}
        for x, y, dist_name in zip(val_a, val_b, dist_names):
            if dist_name == 'W':
                # can normalize the distribution instead
                bin_range = (0, 5)
            else:
                bin_range = (0, 1)
            hist_a, _ = compute_histogram(torch.tensor(x), bin_range=bin_range, density=True, weights=weights)
            hist_b, _ = compute_histogram(torch.tensor(y), bin_range=bin_range, density=True)
            kl_div[dist_name] = self.kl_divergence(hist_b, hist_a) 

        return kl_div 
    
    def calculate_weights(self, probas, low):
        weights = torch.clamp(probas[:, 1], low, 1 - low)
        weights = weights / (1. - weights)
        return weights
    
    def validation_epoch_end(self, val_step_outputs):
        # Log KL Divergence
        weights = torch.cat([row[1] for row in val_step_outputs], dim=0)

        # the first half of the weights are for the first generator
        weights_a = weights[:int(len(weights)/2)] 
        val_dataloader = self.trainer.val_dataloaders[0]
        kl_div = self.compute_kls(val_dataloader.dataset, dist_names=['w', 'x', 'y'], weights=weights_a)
        self.log_dict(kl_div, prog_bar=True)
        
    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch['features'])
        probas = F.softmax(y_hat) 
        weights = self.calculate_weights(probas, 0.001)
        # Compute common step
        loss, labels = self._shared_step(batch, mode='validation')
        return loss, weights

    def test_step(self, batch, batch_idx):
        loss, labels = self._shared_step(batch, mode='test')
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), 3e-5) #3e-5
