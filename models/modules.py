import torch
from torch import nn
from tab_transformer_pytorch import TabTransformer
import math

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


# We dont use this features
@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


class ParticleFlowNetwork(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, input_dims, num_classes,
                 Phi_sizes=(100, 100, 256),
                 F_sizes=(100, 100, 100),
                 use_bn=False,
                 for_inference=False,
                 transform_to_pt=False,
                 **kwargs):
        
        super(ParticleFlowNetwork, self).__init__(**kwargs)
        # input bn
        self.input_bn = nn.BatchNorm1d(input_dims) if use_bn else nn.Identity()
        # per-particle functions
        phi_layers = []
        for i in range(len(Phi_sizes)):
            phi_layers.append(nn.Sequential(
                nn.Conv1d(input_dims if i == 0 else Phi_sizes[i - 1], Phi_sizes[i], kernel_size=1), # this is a linear layer
                nn.BatchNorm1d(Phi_sizes[i]) if use_bn else nn.Identity(),
                nn.ReLU())
            )
        self.phi = nn.Sequential(*phi_layers)
        # global functions
        f_layers = []
        for i in range(len(F_sizes)):
            f_layers.append(nn.Sequential(
                nn.Linear(Phi_sizes[-1] if i == 0 else F_sizes[i - 1], F_sizes[i]),
                nn.ReLU())
            )
        f_layers.append(nn.Linear(F_sizes[-1], num_classes))
        if for_inference:
            f_layers.append(nn.Softmax(dim=1))
        self.fc = nn.Sequential(*f_layers)
        
        # Loss collapses to nan if this is used
        self.transform_to_pt = transform_to_pt 

    def forward(self, features, mask=None):
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        if self.transform_to_pt:
            features = to_ptrapphim(features)
        x = self.input_bn(features)
        x = self.phi(x)

        # hardcoding mask 
        # if mask is not None:
        mask = features[:, 3, :] > 0
        mask = mask.unsqueeze(dim=1)
        x = x * mask.bool().float()
        x = x.sum(-1)
        
        # can optonally add features here before passing it through the F layer
        return self.fc(x) 


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
