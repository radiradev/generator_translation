# Generator Reweighting of DUNE event samples
This library aims to apply ML techniques for event reweighting

<!-- **So, here’s a simple pytorch template that help you get into your main project faster and just focus on your core (Model Architecture, Training Flow, etc)** -->


# Acknowledgments
This work is based on work by Cristovao Vilela - [GeneratorReweight](https://github.com/cvilelahep/GeneratorReweight/)

# Requirements
- [awkard-array](https://github.com/scikit-hep/awkward) (Manipulating nested, variable-sized data using NumPy idioms)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [uproot](https://uproot.readthedocs.io/en/latest/) (Reading ROOT files in Python)
- [pytorch-lgihtning](https://www.pytorchlightning.ai/)(pytorch wrapper)
- [\LaTeX](https://pypi.org/project/latex/)(LaTeX support for plotting utils)

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   
Reweighting distributions using neural networks. More info on the methodology in this [presentation](https://indico.fnal.gov/event/47708/contributions/208129/attachments/139833/175623/cv_generatorrw_20210208.pdf)

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


# In Details
```
├──  notebooks
│    └── data_exploration.ipynb  - a notebook doing some preliminary exploration of the processed data.
│
├──  src
│    └── data_preparation.py - takes root files and converts them into an h5py file
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│
│
├──  models
│   ├── cris_model.py     - this file contains basic nn architecture
│   └── reweighter.py   - a pytorch lightning wrapper for training loops
│
├── main.py             - script that runs training of the project.
```


# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.


