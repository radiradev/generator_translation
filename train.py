from src.root_dataloader import ROOTCLoud
from models.model import LightningModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import argparse
from rich import print
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--n-epochs", type=int, default=2000)
parser.add_argument("--n-workers", type=int, default=2)
parser.add_argument("--root_dir",
                    type=str,
                    default='/eos/home-c/cristova/DUNE/AlternateGenerators/') #/eos/user/r/rradev/particle_cloud/
parser.add_argument('--generator_a', type=str, default='flat_argon_12_GENIEv2')
parser.add_argument('--generator_b', type=str, default='flat_argon_12_GENIEv3_G18_10b')
parser.add_argument('--reload_dataloader_every_n_epochs', type=int, default=1)
args = parser.parse_args()


class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: str):
        """
        Load the val dataset once but we load the train dataset at each epoch
        to go through all train files.
        """
        self.val_dataset = ROOTCLoud(
            data_dir = args.root_dir, 
            generator_a=args.generator_a, 
            generator_b=args.generator_b, 
            shuffle_data=False, 
            validation=True)

    def train_dataloader(self):
        train_dataset = ROOTCLoud(
            data_dir = args.root_dir, 
            generator_a=args.generator_a, 
            generator_b=args.generator_b, 
            shuffle_data=True, 
            validation=False)
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=False)


data = DataModule()
# Init our model
model = LightningModel().float()



# Create a callback
checkpoint_callback = ModelCheckpoint(save_top_k=10,
                                      verbose=True,
                                      monitor='validation_f1_score',
                                      mode='max')
progress_bar = TQDMProgressBar(refresh_rate=250)

# Initialize a trainer
trainer = Trainer(
    default_root_dir=f'lightning_logs/{args.generator_a} and {args.generator_b}',
    callbacks=[checkpoint_callback, progress_bar],
    accelerator='gpu',
    devices=1,
    max_epochs=args.n_epochs,
    log_every_n_steps=1000,
    reload_dataloaders_every_n_epochs=args.reload_dataloader_every_n_epochs,
    check_val_every_n_epoch=10,
    fast_dev_run=False,
    num_sanity_val_steps=-1 # run a full validation epoch before starting training
    
    # max_steps = 100000,
    # default_root_dir=args.root_dir,
)

#  Train the model ⚡
trainer.fit(model, data)
