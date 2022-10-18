import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Adam
from torchmetrics.functional import accuracy, f1_score

from models.modules import ParticleFlowNetwork
from src.utils.funcs import compute_histogram, detach_tensor


class LightningModel(pl.LightningModule):

    def __init__(self, learning_rate=None, batch_size=None, transform_to_pt=True, use_embeddings=True, input_dims=5, input_bn=False):
        super().__init__()
        self.batch_size = batch_size
        # self.lr = hparams.lr
        self.learning_rate = learning_rate
        # self.save_hyperparameters(learning_rate)
        self.net = ParticleFlowNetwork(input_dims, num_classes=2, transform_to_pt=transform_to_pt, Phi_sizes=(100, 100 , 128), use_embeddings=use_embeddings, input_bn=input_bn)
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
    
    def get_plotly_equivalent(self, hist):
        bins, edges = detach_tensor(hist[0]), detach_tensor(hist[1])
        left,right = edges[:-1],edges[1:]
        X = np.array([left,right]).T.flatten()
        Y = np.array([bins,bins]).T.flatten()
        return X, Y

    def log_plotly(self, hist_a, hist_b, hist_reweighted, dist_name):
        
        kl_div = self.kl_divergence(hist_b[0], hist_a[0])
        kl_div_reweighted = self.kl_divergence(hist_reweighted[0], hist_a[0])
        x1, x2 = self.get_plotly_equivalent(hist_a), self.get_plotly_equivalent(hist_b)
        x3 = self.get_plotly_equivalent(hist_reweighted)
        fig = go.Figure(data=[
            go.Line(name=f'nominal, kl: {kl_div:.2f}', x=x1[0], y=x1[1]),
            go.Line(name=f'target', x=x2[0], y=x2[1]),
            go.Line(name=f'reweighted, kl {kl_div_reweighted:.2f}', x=x3[0], y=x3[1])])

        fig.update_layout(
            barmode='overlay',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01)
            margin=go.layout.Margin(
                l=0,
                r=0,
                b=0,
                t=0)
        )
        fig.update_traces(marker=dict(line=dict(width=0,
                                                color='red')))
        logger = self.trainer.logger.experiment
        logger.log({dist_name : fig})
    
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
        for x, y, dist_name in zip(val_a, val_b, dist_names):
            if dist_name == 'w':
                # can normalize the distribution instead
                bin_range = (0, 5)
            else:
                bin_range = (0, 1)
            
            nominal = compute_histogram(torch.tensor(x), bin_range=bin_range, density=True, weights=weights)
            reweighted = compute_histogram(torch.tensor(x), bin_range=bin_range, density=True, weights=weights)
            target = compute_histogram(torch.tensor(y), bin_range=bin_range, density=True)
            
            self.log_plotly(nominal, target, reweighted, dist_name)
        
    
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
        self.compute_kls(val_dataloader.dataset, dist_names=['w', 'x', 'y'], weights=weights_a)
        
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
