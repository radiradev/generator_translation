import torch
from tab_transformer_pytorch import TabTransformer

cont_mean_std = torch.randn(10, 2)

tab_transformer = TabTransformer(
    categories = (2, 2, 2, 2, 2),      # tuple containing the number of unique values within each category
    num_continuous = 1,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = torch.nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    # continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
)

class CrisModel(torch.nn.Module):
    def __init__(self):
        super(CrisModel, self).__init__()
        self._nn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(23),
            torch.nn.Linear(23, 64), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,64), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,64), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,64), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,1)
            )
        
    def forward (self, x):
        return self._nn(x)
