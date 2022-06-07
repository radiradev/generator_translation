from data.dataset import NuDataset
from models.reweighter import Reweighter
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

#Load data
HOME_DIR = '/eos/home-r/rradev/generator_reweigthing/'
GENIE_DIR = 'argon_GENIEv2.h5'
NEUT_DIR = 'argon_NEUT.h5'
BATCH_SIZE = 512
NUM_WORKERS = 4

data_dirs = [HOME_DIR + GENIE_DIR, HOME_DIR + NEUT_DIR]
train_ds = NuDataset(data_dirs, test=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

# Init our model
mnist_model = Reweighter()

# Initialize a trainer
trainer = Trainer(
    gpus=1,
    max_epochs=3,
    progress_bar_refresh_rate=20,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)