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
-  [Project Structure](#project-structure)
-  [Future Work](#future-work)

# In a Nutshell   
Reweighting distributions using neural networks. More info on the methodology in this [presentation](https://indico.fnal.gov/event/47708/contributions/208129/attachments/139833/175623/cv_generatorrw_20210208.pdf)


# Project structure


```
├── models
│   ├── model.py            - pytorch lightning wrapper to train the model.
│   ├── modules.py          - NN architectures to be used in the model. 
├── notebooks
│   ├── reweighting.ipynb   - an analysis notebook of reweighting done with this framework.
│   └── toy_example.ipynb   - an example how reweighting is done using NNs
├── src
│   ├── root_dataloader.py  - pytorch dataloader that loads a ROOT file directly or from a numpy array
│   ├── to_numpy.py         - converts ROOT file to numpy
│   └── utils
│       ├── funcs.py        - various utility functions. 
│       ├── plotting.py     - utilies for plotting distributions and probability plots.
├── tests
│   └── test_gibuu.py       - test loading of the GIBUU data.
├── train.py                - script that runs training of the project.
└── reweight.py             - generates a reweighted sample using the trained model.
```
# Future Work


