import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import uproot 
import vector
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report
from models.model import LightningModel

from src.root_dataloader import pad_array, rec2array
from src.utils.funcs import rootfile_to_array, detach_tensor, get_vars_meta
from src.utils.plotting import plot_distribution, probability_plots


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir", type=str, default="/eos/home-c/cristova/DUNE/AlternateGenerators/"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="/data/rradev/generator_reweight/lightning_logs/GENIEv2 and GENIEv3/lightning_logs/version_16/checkpoints/epoch=327-step=1025000.ckpt",
)
args = parser.parse_args()

def make_plots(nominal, other, predicted_weights, vars_meta, figsize=None):
    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize

    # Create variable names    
    col_names, n_bins, x_min, x_max, fig_title = vars_meta

    # Iterate through variables and plot
    for idx, variable in enumerate(col_names):
        hist_range = (float(x_min[idx]), float(x_max[idx]))
        plot_distribution(
            nominal[:, idx],
            other[:, idx],
            predicted_weights,
            n_bins=int(n_bins[idx]),
            errorbars=False,
            density=True,
            range=hist_range,
            label_nominal="GENIEv2",
            label_target="GENIEv3",
            ratio_limits=(0.8, 1.2),
        )
        plt.suptitle(variable)
        plt.savefig(f"saved_plots/{variable}.png")

def get_p4(filename, max_len=30):
    with uproot.open(filename) as f:
        tree = f['FlatTree_VARS']
        px = tree['px']
        py = tree['py']
        pz = tree['pz']
        energy = tree['E']

        p4 = vector.zip({
            'px': px.array(),
            'py': py.array(),
            'pz': pz.array(), 
            'E': energy.array()
        })

        X = pad_array(p4, max_len, axis=1) 
        X = rec2array(X).swapaxes(1, 2)
        return torch.tensor(X)

def calculate_weights(probas, low):
    weights = np.clip(probas[:, 1], low, 1 - low)
    weights = weights / (1. - weights)
    weights = np.squeeze(np.nan_to_num(weights))
    return weights

# Load model with weights (should be automatic ie dont hardspecify the argument)
print("Loading model from checkpoint")
model = LightningModel(hparams=args).load_from_checkpoint(args.checkpoint_path)
model.eval()

v2_filename = args.root_dir + 'flat_argon_12_GENIEv2_1M_005_NUISFLAT.root'
v3_filename = args.root_dir + 'flat_argon_12_GENIEv3_G18_10b_00_000_1M_007_NUISFLAT.root'

p4_nominal = get_p4(v2_filename)
p4_target = get_p4(v3_filename)

def predict_weights(p4):
    batch_size = 1000
    n_iterations = int(len(p4)/batch_size)
    weights_list = []
    probas_list = []
    with torch.no_grad():
        for i in tqdm(range(n_iterations)):
            # Indices to loow through model
            idx_low = i * batch_size
            idx_high = idx_low + batch_size

            # Get predictions
            y_hat = model(p4[idx_low:idx_high])
            # Get weights
            probas = F.softmax(y_hat)        
            weights = calculate_weights(probas, low=0.0001)
            weights_list.append(weights)
            probas_list.append(probas)
        weights = np.hstack(weights_list)
        probas = np.vstack(probas_list)
        print(probas.shape, weights.shape)
    return weights, probas

weights, probas_nominal = predict_weights(p4_nominal)
_, probas_target = predict_weights(p4_target)

plt.hist(weights, bins=100)
plt.yscale('log')
plt.savefig(f"saved_plots/hist.png")

plt.rcParams['figure.figsize'] = (20, 20)
probability_plots(probas_nominal[:, 1], probas_target[:, 1])
plt.savefig('saved_plots/probability.png')

nominal, nominal_weights = rootfile_to_array(v2_filename)
target, target_weights = rootfile_to_array(v3_filename)

figsize = (12, 10)
many_bins = 100
vars_meta = get_vars_meta(many_bins)
make_plots(nominal, target, weights, vars_meta, figsize)

# GENIEv2 label - 0 
label = np.vstack([np.zeros_like(nominal_weights), np.ones_like(target_weights)])
probas = np.hstack([probas_target[:, 1], probas_nominal[:, 1]])
print(probas.shape)
pred = (probas < 0.5).astype(int)

clf_report = classification_report(
    y_true=label,
    y_pred=pred)
print(clf_report)



