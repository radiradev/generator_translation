import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib import rc

# Define default plot styles
plot_style_0 = {'histtype':'step', 'color':'black', 'linewidth':2, 'linestyle':'--'}
plot_style_1 = {'alpha':0.5}

# Absolute plotting params
rc('font', family='serif')
rc('font', size=22) 
rc('xtick', labelsize=15) 
rc('ytick', labelsize=15) 
rc('legend', fontsize=15)
rc('figure', figsize=(15, 10))



def plot_distribution(nominal, target, weights, n_bins=None, range=None, errorbars=False, density=True):
    if range is None:
        min_val = 0
        max_val = np.min([
            np.max(nominal), 
            np.max(target)])
        range = (min_val, max_val)

    fig, axs = plt.subplots(2, 1)

    # Histograms
    counts_ref, bins_ref, _ = axs[0].hist(target, label='Target', bins=n_bins, **plot_style_0, range=range, density=density)
    counts_nominal, _, _ = axs[0].hist(nominal, label='Nominal', bins=n_bins,  **plot_style_1, range=range, density=density)
    counts_reweighted, _, _ = axs[0].hist(nominal, weights=weights, bins=n_bins, label='Nominal to Target', **plot_style_1, range=range, density=density)
    axs[0].legend()

    # Ratio Plots
    ratio_plot(bins_ref, counts_ref, counts_nominal, axs, errorbars, color='blue')
    ratio_plot(bins_ref, counts_ref, counts_reweighted, axs, errorbars, color='orange')

def ratio_plot(bins_ref, counts_ref, counts, axs, errorbars, color):
    x_values, y_values, y_errors = ratios(bins_ref, counts_ref, counts)
    axs[1].plot(x_values, np.ones_like(counts_ref), '--')
    if errorbars:
        axs[1].errorbar(x=x_values, y=y_values, yerr=y_errors, fmt='o', color=color)
        axs[1].set_ylim(0.5, 1.5)
    else:
        axs[1].plot(x_values, y_values, 'o', color=color)
        axs[1].set_ylim(0.5, 1.5)
    axs[1].set_title('Nominal/Target')


def ratios(bins_ref, counts_ref, counts):
    x_values = [0.5*(bins_ref[i+1]+bins_ref[i]) for i in range(len(counts_ref))]
    
    y_values = (counts + 1e-9)/(counts_ref + 1e-9)
    y_errors = np.sqrt(counts)/(counts + 1e-9) \
             + np.sqrt(counts_ref)/(counts_ref + 1e-9)
    y_errors *= y_values
    
    return x_values, y_values, y_errors

    