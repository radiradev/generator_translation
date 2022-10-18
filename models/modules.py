import math

import numpy as np
import torch
from torch import nn

from models.integer_embedding import IntegerEmbedding
from src.utils.funcs import get_pdg_codes


def ids_to_pdg(id):
    pdg_codes = get_pdg_codes()
    to_values = np.arange(1, len(pdg_codes) + 1)
    map_id_to_pdg = dict(zip(pdg_codes, to_values))
    return map_id_to_pdg[id]

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

def to_pabs_phi_theta(x, return_energy=True, eps=1e-8,):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1, 1), dim=1)
    p_absolute = torch.log(torch.sqrt(to_pt2(x))).clamp(min=1e-20)
    theta = torch.arctan2(px, pz)/(math.pi)
    phi = to_phi(px, py)/(math.pi)
    if return_energy:
       return torch.cat((p_absolute, theta, phi, energy/torch.tensor(50.0)), dim=1)
    else:
       return torch.cat((p_absolute, theta, phi), dim=1)


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
                 use_embeddings=True,
                 num_particle_types=21,
                 max_num_protons = 12, 
                 max_num_neutrons = 12,
                 input_bn=False,
                 **kwargs):
        
        super(ParticleFlowNetwork, self).__init__(**kwargs)
        # input bn
        self.input_bn = nn.BatchNorm1d(input_dims) if use_bn else nn.Identity()
        
        # Input embedding for particle class
        embedding_depth = int(num_particle_types / 2)
        self.type_embeddings = nn.Embedding(num_particle_types, embedding_depth) if use_embeddings else nn.Identity()
        self.proton_embeddings = IntegerEmbedding(max_num_protons, 5)
        self.neutron_embeddings = IntegerEmbedding(max_num_neutrons, 5)
        if use_embeddings:
            input_dims += embedding_depth - 1 # -1 if it does not use energy
        
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
            # Adding + 10 for particle embedding depth
            f_layers.append(nn.Sequential(
                nn.Linear(Phi_sizes[-1] + 2 if i == 0 else F_sizes[i - 1], F_sizes[i]),
                nn.ReLU())
            )
        f_layers.append(nn.Linear(F_sizes[-1], num_classes))
        if for_inference:
            f_layers.append(nn.Softmax(dim=1))
        self.fc = nn.Sequential(*f_layers)
        
        self.transform_to_pt = transform_to_pt 

    def forward(self, features):
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        
        p4, particle_type = features[:, :-1, :], features[:, -1, :]
        mask = p4[:, -1, :] > 0 # Energy component has to be bigger than 0

        count_protons = torch.sum(particle_type == ids_to_pdg(2212), axis=1).to(torch.long)
        count_neutrons =  torch.sum(particle_type == ids_to_pdg(2112), axis=1).to(torch.long)
        
        type_embeddings = self.type_embeddings(particle_type.to(torch.long))
        proton_embeddings = count_protons.clamp(max=10)/10.0
        neutron_embeddings = count_neutrons.clamp(max=10)/10.0
        counts = torch.stack((proton_embeddings, neutron_embeddings), dim=1)
        
        if self.transform_to_pt:
            p4 = to_pabs_phi_theta(p4)
        
        x = torch.cat((p4, torch.swapaxes(type_embeddings, 1,2)), dim=1)
        x = self.input_bn(x)
        x = self.phi(x.to(torch.float32))
        mask = mask.unsqueeze(dim=1)
        x = x * mask.bool().float()
        x = x.sum(-1)
        
        x = torch.cat((x, counts), dim=1)
        # can optonally add features here before passing it through the F layer
        return self.fc(x) 
