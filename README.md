# Generator Reweighting of DUNE event samples
This library aims to apply ML techniques for event reweighting

<!-- **So, here’s a simple pytorch template that help you get into your main project faster and just focus on your core (Model Architecture, Training Flow, etc)** -->


# Acknowledgments
This work is based on work by Cristovao Vilela - [GeneratorReweight](https://github.com/cvilelahep/GeneratorReweight/)

# Requirements
- [awkard-array](https://github.com/scikit-hep/awkward) (Manipulating nested, variable-sized data using NumPy idioms)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [uproot](https://uproot.readthedocs.io/en/latest/) (Reading ROOT files in Python)

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   
We are exploring the use of optimal transport techniques combined with a generative model to **reweigth** event samples. 
- The optimal transport follows the implementation of [Input Convex Neural Newtorks](https://github.com/cspollard/picnn) with the corresponding article of [Transport away your problems](https://www.sciencedirect.com/science/article/pii/S0168900221010020?via%3Dihub). 

<!-- ```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

   
- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
``` -->

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**


# In Details
```
├──  notebooks
│    └── data_exploration.ipynb  - a notebook doing some preliminary exploration of the processed data.
│
│
├──  src
│    └── data_preparation.py - takes root files and converts them into an h5py file
│
```
<!-- 
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
│
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
│   
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
``` -->


# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.


