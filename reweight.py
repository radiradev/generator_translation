import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import uproot 
import vector
import torch.nn.functional as F
from tqdm import tqdm
import glob
import awkward as ak
from sklearn.metrics import classification_report
from models.model import LightningModel

from src.root_dataloader import pad_array, rec2array
from src.utils.funcs import rootfile_to_array, detach_tensor, get_vars_meta, map_to_integer
from src.utils.plotting import plot_distribution, probability_plots

def get_pdg_codes():
    leptons = [11, -11, 13, -13, 15, -15]
    neutrinos = [15, -15, 12, -12]
    hadrons = [2212, 2112]
    pions = [211, -211, 111]
    kaons = [321, -321, 311, 130, 310]
    return leptons + neutrinos + hadrons + pions + kaons

@np.vectorize
def map_array(val, dictionary):
    return dictionary[val] if val in dictionary else 0 

def to_ids(pdg_array):
    pdg_codes = get_pdg_codes()
    to_values = np.arange(1, len(pdg_codes) + 1)
    dict_map = dict(zip(pdg_codes, to_values))
    flat_pdg = ak.flatten(pdg_array)
    ids = map_array(flat_pdg, dict_map)
    counts = ak.num(pdg_array)
    return ak.unflatten(ids, counts)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir", type=str, default="/eos/home-c/cristova/DUNE/AlternateGenerators/"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default='/data/rradev/generator_reweight/lightning_logs/flat_argon_12_GENIEv2 and flat_argon_12_GENIEv3_G18_10b/lightning_logs/version_9/checkpoints/epoch=1999-step=7814000.ckpt',
)
args = parser.parse_args()


def get_last_run(generator_a, generator_b):
    return glob.glob(f'lightning_logs/{generator_a} and {generator_b}/lightning_logs/*')[-1]

def get_last_saved_checkpoint(last_run):
    return glob.glob(f'{last_run}/checkpoints/*')[-1]

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
        if 'is' not in variable:
            plt.savefig(f"saved_plots/{variable}.png")
        else:
            plt.show()

def get_p4(filename, max_len=30):
    print('Reading variables')
    with uproot.open(filename) as f:
        tree = f['FlatTree_VARS']
        px = tree['px']
        py = tree['py']
        pz = tree['pz']
        energy = tree['E']
        particle_id = tree['pdg']

        p4 = ak.zip({
            'px': px.array(),
            'py': py.array(),
            'pz': pz.array(), 
            'E': energy.array(),
            'particle_id': to_ids(particle_id.array())
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

generator_a = 'flat_argon_12_GENIEv2'
generator_b = 'flat_argon_12_GENIEv3_G18_10b'

if args.checkpoint_path is None:
    last_run = get_last_run(generator_a, generator_b)
    args.checkpoint_path = get_last_saved_checkpoint(last_run)
    print(f'Reweighting using {args.checkpoint_path}')

model = LightningModel(transform_to_pt=False, use_embeddings=False, input_dims=4)
checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()


v2_filename = args.root_dir + f'{generator_a}_1M_04*_NUISFLAT.root'
v3_filename = args.root_dir + f'{generator_b}*1M_04*_NUISFLAT.root'

def predict_weights(filename, low):
    filenames = glob.glob(filename)
    batch_size = 1000
    weights_list = []
    probas_list = []
    
    with torch.no_grad():
        for filename in filenames:
            p4 = get_p4(filename)
            n_iterations = int(len(p4)/batch_size)
            for i in tqdm(range(n_iterations)):
                
                # Indices to low through model
                idx_low = i * batch_size
                idx_high = idx_low + batch_size

                # Get predictions
                y_hat = model(p4[idx_low:idx_high])
                # Get weights
                probas = F.softmax(y_hat)        
                weights = calculate_weights(probas, low=low)
                weights_list.append(weights)
                probas_list.append(probas)
        
        # After iterating through create weight files
        weights = np.hstack(weights_list)
        probas = np.vstack(probas_list)
        print(probas.shape, weights.shape)
    return weights, probas

weights, probas_nominal = predict_weights(v2_filename, low=0.001)
_, probas_target = predict_weights(v3_filename, low=0.001)


nominal = np.vstack([rootfile_to_array(filename) for filename in glob.glob(v2_filename)])
target = np.vstack([rootfile_to_array(filename) for filename in glob.glob(v3_filename)])

plt.hist(weights, bins=100)
plt.yscale('log')
plt.savefig(f"saved_plots/hist.png")

figsize = (12, 10)
many_bins = 100
vars_meta = get_vars_meta(many_bins)
make_plots(nominal, target, weights, vars_meta, figsize)

plt.rcParams['figure.figsize'] = (20, 20)
probability_plots(probas_nominal[:, 1], probas_target[:, 1])
plt.savefig('saved_plots/probability.png')


# GENIEv2 label - 0 
label = np.vstack([np.zeros(shape=weights.shape[0]), np.ones_like(shape=weights.shape[0])])
probas = np.hstack([probas_target[:, 1], probas_nominal[:, 1]])
print(probas.shape)
pred = (probas < 0.5).astype(int)

clf_report = classification_report(
    y_true=label,
    y_pred=pred)
print(clf_report)



