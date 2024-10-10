### Imports ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

MOTIF_COLS = ['tf', 'gene', 'edge']

### Functions ###
def load_prior(
    path: Path,
) -> pd.DataFrame:
    """
    Loads a motif prior file into a pandas DataFrame

    Parameters
    ----------
    path : Path
        Path to the motif prior

    Returns
    -------
    pd.DataFrame
        The processed pandas DataFrame
    """

    return pd.read_csv(path, sep='\t', header=None, names=MOTIF_COLS)


def describe_prior(
    prior: pd.DataFrame,
) -> None:
    """
    Provides summary statistics about a motif prior: numbers of TFs,
    genes, edges, and density.

    Parameters
    ----------
    prior : pd.DataFrame
        Pandas DataFrame with the motif prior
    """

    n_tfs = prior['tf'].nunique()
    n_genes = prior['gene'].nunique()
    n_edges = len(prior)

    print ('Number of unique TFs:', n_tfs)
    print ('Number of unique genes:', n_genes)
    print ('Number of edges:', n_edges)
    print (f'Network density: {100 * n_edges / (n_tfs * n_genes):.2f} %')


def plot_confusion_matrix(
    data: np.array,
) -> plt.Axes:
    """
    Plots a confusion matrix for two motif priors.

    Parameters
    ----------
    data : np.array
        Calculated confusion matrix

    Returns
    -------
    plt.Axes
        Matplotlib Axes with the newly created figure
    """

    s_data = data * 100 / np.sum(data)

    fig,ax = plt.subplots(figsize=(6,6))
    mappable = ax.imshow(s_data, cmap='Blues', vmin=0, vmax=100)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.8)
    cb.ax.tick_params(labelsize=14)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticks([0,1], labels=['0 in first prior', '1 in first prior'],
        fontsize=14)
    ax.set_yticks([0,1], labels=['0 in second prior', '1 in second prior'],
        rotation='vertical', va='center', fontsize=14)
    for i in range(2):
        for j in range(2):
            ax.text(i, j, f'{data[i][j]:,d}\n{s_data[i][j]:.2f} %',
                ha='center', va='center', fontsize=16,
                c='white' if s_data[i][j] > 50 else 'black')

    return ax


def compare_priors(
    prior_1: pd.DataFrame,
    prior_2: pd.DataFrame,
) -> plt.Axes:
    """
    Compares two motif priors. Reports summary statistics for both, then
    compares the density of their common subset, also provides a
    classification report and plots a confusion matrix.

    Parameters
    ----------
    prior_1 : pd.DataFrame
        Pandas DataFrame with the first prior
    prior_2 : pd.DataFrame
        Pandas DataFrame with the second prior

    Returns
    -------
    plt.Axes
        Matplotlib Axes with the confusion matrix
    """

    print ('Statistics for the first prior:')
    describe_prior(prior_1)
    print ()
    print ('Statistics for the second prior:')
    describe_prior(prior_2)

    common_tfs = set(prior_1['tf'].unique()).intersection(
        prior_2['tf'].unique())
    common_genes = set(prior_1['gene'].unique()).intersection(
        prior_2['gene'].unique())
    print ()
    print ('Number of common TFs:', len(common_tfs))
    print ('Number of common genes:', len(common_genes))

    common_index = pd.MultiIndex.from_product([sorted(common_tfs),
        sorted(common_genes)])
    prior_1_mod = prior_1.set_index(['tf', 'gene']).reindex(
        common_index, fill_value=0)
    prior_2_mod = prior_2.set_index(['tf', 'gene']).reindex(
        common_index, fill_value=0)
    comp_df = prior_1_mod.join(prior_2_mod, lsuffix='_1', rsuffix='_2')

    print ()
    print ('Network density in common TF/genes for the first prior:',
        f'{100 * comp_df["edge_1"].mean():.2f} %')
    print ('Network density in common TF/genes for the second prior:',
        f'{100 * comp_df["edge_2"].mean():.2f} %')
    print ()
    print (classification_report(comp_df['edge_1'], comp_df['edge_2']))

    return plot_confusion_matrix(
        confusion_matrix(comp_df['edge_1'], comp_df['edge_2']))