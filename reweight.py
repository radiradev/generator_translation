from array import array
from src.root_dataloader import NumPyDataset
from models.model import LightningModel
import argparse
import torch
import numpy as np
from utils.funcs import get_constants, detach_tensor
import pandas as pd


def replace_nan_values(array):
    """
    Replaces nan values within an array
    """
    X = array.copy()
    nan_index = np.isnan(X)
    X[nan_index] = np.random.randn(len(X[nan_index]))
    return X


def calculate_logits(generator, model):
    """
    Returns pandas df with logits and features for a given generator dataset
    """
    # Get column names for pandas df
    _, col_names, _ = get_constants()
    col_names.append('labels')

    # Calculate model output
    with torch.no_grad():
        batch = replace_nan_values(dataset.load_generator(generator))
        print(f'Calculating logits for {generator}')
        features = torch.tensor(batch[:, :-1], dtype=torch.float)
        logits = detach_tensor(model(features))

        # Save files
        array = np.concatenate((batch, logits), axis=1)

    return array


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir",
                    type=str,
                    default='/eos/user/r/rradev/generator_reweigthing/')
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=
    '/data/rradev/generator_reweight/lightning_logs/to_neut2/checkpoints/epoch=47-step=37536.ckpt'
)
args = parser.parse_args()

# Load model with weights
print('Loading model from checkpoint')
model = LightningModel(hparams=args).load_from_checkpoint(args.checkpoint_path)
model.eval()

# Load dataset
generator_a = 'GENIEv2'
generator_b = 'NEUT'
number_of_files = 1
print(f'Loading dataset with target generator {generator_b}')

dataset = NumPyDataset(args.root_dir,
                      generator_a=generator_a,
                      generator_b=generator_b,
                      shuffle=False,
                      number_of_files = number_of_files)

arr = np.concatenate([calculate_logits(generator_a, model), calculate_logits(generator_b, model)])
np.save(f'reweighted_samples/{generator_a}_to_{generator_b}.npy', arr)
