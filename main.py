from src.root_dataloader import ROOTDataset
from models.reweighter import Reweighter
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--n-epochs", type=int, default=10)
parser.add_argument("--n-workers", type=int, default=2)
parser.add_argument("--root_dir",
                    type=str,
                    default='/eos/home-c/cristova/DUNE/AlternateGenerators/')
parser.add_argument('--lr', type=int, default=1e-4)
parser.add_argument('--buffer_size', type=int, default=10e4)
args = parser.parse_args()

# Load Data
dataset = ROOTDataset(args.root_dir)

# Split into val and train
val_dataset_len = int(len(dataset) * 0.2)
train_dataset_len = len(dataset) - val_dataset_len
train_dataset, val_dataset = random_split(
    dataset, lengths=[train_dataset_len, val_dataset_len])

train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          num_workers=args.n_workers,
                          shuffle=True)
val_loader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.n_workers,
                        shuffle=False)

# Init our model
model = Reweighter(hparams=args).float()

# Create a callback
checkpoint_callback = ModelCheckpoint(save_top_k=10,
                                      verbose=True,
                                      monitor='validation_acc',
                                      mode='max')

# Initialize a trainer
trainer = Trainer(
    callbacks=checkpoint_callback,
    gpus=1,
    max_epochs=args.n_epochs,
    progress_bar_refresh_rate=100,
    log_every_n_steps=500,
    # max_steps = 100000,
    # default_root_dir=args.root_dir,
)

# Train the model ⚡
trainer.fit(model, train_loader, val_loader)
