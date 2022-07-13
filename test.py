from src.root_dataloader import ROOTDataset
from models.reweighter import Reweighter
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import torch
import numpy as np

def replace_nan_values(array):
    X = array.copy()
    nan_index = np.isnan(X)
    X[nan_index] = np.random.randn(len(X[nan_index]))
    return X

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--n-epochs", type=int, default=10)
parser.add_argument("--n-workers", type=int, default=2)
parser.add_argument("--root_dir",
                    type=str,
                    default='/eos/home-c/cristova/DUNE/AlternateGenerators/')
parser.add_argument('--lr', type=int, default=1e-4)
parser.add_argument("--checkpoint_path", type=str, default='/data/rradev/generator_reweight/lightning_logs/version_12/checkpoints/epoch=7-step=50000.ckpt')
args = parser.parse_args()

# Load model with weights
model = Reweighter(hparams=args).load_from_checkpoint(args.checkpoint_path)
model.eval()

# Load dataset
dataset = ROOTDataset(args.root_dir, preload_data=False, shuffle=False)

# Get predictions
with torch.no_grad():
    for generator in ['GENIEv2', 'NUWRO']:
        # Load data
        batch = replace_nan_values(dataset.load_generator(generator))

        # Last column is the label
        features = torch.tensor(batch[:, :-1], dtype=torch.float) #last column is the label
        print(f'Calculating logits for generator')
        logits = model(features).detach().cpu().numpy()
        features = features.detach().cpu().numpy()

        # Save files
        np.save(f'reweighted_samples/{generator}_logits.npy', logits)
        np.save(f'reweighted_samples/{generator}_features.npy', features)
