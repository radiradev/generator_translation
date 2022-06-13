from data.dataset import NuDataset, build_datapipes
from models.reweighter import Reweighter
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--n-epochs", type=int, default=5)
parser.add_argument("--n-workers", type=int, default=4)
parser.add_argument("--root_dir", type=str, default='/eos/home-r/rradev/generator_reweigthing/')
args = parser.parse_args()

# Load Data
train_datapipe = build_datapipes(args.root_dir, mode='train')
val_datapipe = build_datapipes(args.root_dir, mode='val')

train_loader = DataLoader(train_datapipe, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
val_loader = DataLoader(val_datapipe, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)

# Init our model
model = Reweighter(hparams=args)

# Initialize a trainer
trainer = Trainer(
    gpus=1,
    max_epochs=args.n_epochs,
    progress_bar_refresh_rate=20,
)

# Train the model ⚡
trainer.fit(model, train_loader, val_loader)