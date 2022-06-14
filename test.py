from data.dataset import NuDataset, build_datapipes
from models.reweighter import Reweighter
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import argparse
import torch
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=int(10e3))
parser.add_argument("--n-epochs", type=int, default=5)
parser.add_argument("--n-workers", type=int, default=3)
parser.add_argument("--root_dir", type=str, default='/eos/home-r/rradev/generator_reweigthing/')
parser.add_argument("--checkpoint_path", type=str, default='/data/rradev/generator_reweight/lightning_logs/version_1/checkpoints/epoch=0-step=1804689.ckpt')
args = parser.parse_args()

# Load model with weights
model = Reweighter(hparams=args).load_from_checkpoint(args.checkpoint_path)


samples = ['GENIEv2_test', 'NEUT_test']

for sample in samples:
    # Load data for the sample
    test_datapipe = build_datapipes(args.root_dir, mode=sample, buffer_size=10e4)
    test_loader = DataLoader(test_datapipe, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)

    # Predict probabilities
    batch = next(iter(test_loader))
    logits = model(batch['features'])
    probas = torch.sigmoid(logits)
    
    torch.save(probas, f'reweighted_samples/{sample}_probas.pt')
    torch.save(batch['features'], f'reweighted_samples/{sample}_features.pt')
