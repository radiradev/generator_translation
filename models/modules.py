import torch
from torch import nn
import math

def to_pt2(x, eps=1e-8):
    pt2 = x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2

def to_phi(px, py, eps=1e-8):
    lepton_px = px[0]
    lepton_py = py[0]
    cross = lepton_px * py - lepton_py * px
    dot = lepton_px * px + lepton_py * py
    return torch.atan2(dot, cross)

def to_pabs_phi_theta(x, return_pid=False, eps=1e-8,):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    p_absolute = torch.log(torch.sqrt(to_pt2(x))).clamp(min=1e-20)
    theta = torch.arctan2(px, pz)/(math.pi)
    phi = to_phi(px, py)/(math.pi)
    pid = 0
    if not return_pid:
       return torch.cat((p_absolute, theta, phi, energy/torch.tensor(50.0)), dim=1)
    else:
       return torch.cat((p_absolute, theta, phi, pid), dim=1)


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
                 transform_to_pt=True,
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
        
        self.transform_to_pt = transform_to_pt 

    def forward(self, features, mask=None):
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        mask = features[:, 3, :] > 0 # Energy component has to be bigger than 0
        if self.transform_to_pt:
            features = to_pabs_phi_theta(features)
        x = self.input_bn(features)
        x = self.phi(x)
        mask = mask.unsqueeze(dim=1)
        x = x * mask.bool().float()
        x = x.sum(-1)
        
        # can optonally add features here before passing it through the F layer
        return self.fc(x) 