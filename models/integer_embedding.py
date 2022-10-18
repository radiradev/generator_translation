import torch
from torch import nn


class IntegerEmbedding(nn.Module):
    def __init__(self, max_integer, embedding_dim, swap_axes=False):
        """Embedding layer suited for integer data. The embedding
        vector is the same for all integers > max_integer.

        Args:
            max_integer (int): Maximum expected integer in our column
            embedding_dim (int): The embedding depth of nn.Embedding 
        """
        super().__init__()
        self.max_integer = max_integer + 1
        self.swap_axes = swap_axes
        self.embedding = nn.Embedding(self.max_integer, embedding_dim)
    
    def __call__(self, x):
        x = x.clip(0, self.max_integer - 1)
        if self.swap_axes:
            torch.swapaxes(x, 1, 2)
        return self.embedding(x)
        