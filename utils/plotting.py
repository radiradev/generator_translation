import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib import rc

# Define default plot styles
plot_style_0 = {'histtype':'step', 'color':'black', 'linewidth':2, 'linestyle':'--', 'density':True}
plot_style_1 = {'alpha':0.5, 'density':True}

# Absolute plotting params
rc('font', family='serif')
rc('font', size=22) 
rc('xtick', labelsize=15) 
rc('ytick', labelsize=15) 
rc('legend', fontsize=15)
rc('figure', figsize=(15, 10))


def plot_distribution(nominal, target, weights, n_bins=50, range=None):
    if range is None:
        min_val = 0
        max_val = np.min([
            np.max(nominal), 
            np.max(target)])
        range = (min_val, max_val)

    plt.hist(target, label='Target', bins=n_bins, **plot_style_0, range=range)
    plt.hist(nominal, label='Nominal', bins=n_bins,  **plot_style_1, range=range)
    plt.hist(nominal, weights=weights, bins=n_bins, label='Nominal to Target', **plot_style_1, range=range)
    plt.legend()

