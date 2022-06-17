import torch

class CrisModel(torch.nn.Module):
    def __init__(self):
        super(CrisModel, self).__init__()
        self._nn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(23),
            torch.nn.Linear(23, 64), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,64), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            # torch.nn.Linear(64,64), torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(64),
            # torch.nn.Linear(64,64), torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(64),
            # torch.nn.Dropout(0.8),
            torch.nn.Linear(64,1)
            )
        
    def forward (self, x):
        return self._nn(x)
