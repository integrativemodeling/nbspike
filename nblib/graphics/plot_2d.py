"""
Utility functions for 2D plots.
"""

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

import matplotlib ; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .. import utils


def plot_all_XL_distance_histogram(histogram_files, outfile,
                                   ligand_colors, cutoff=28.0):
    """
    Plot distributions of crosslinked residue distances.
    
    Args:
    histogram_files (list): List of ASCII (.txt) files for distance
    distributions for each receptor-ligand crosslink dataset.
    Each histogram file has two columns: (bin center, bin count) 
    
    outfile (str): Output file name (.png preferred).
    
    ligand_colors (str): Line-color to use for each ligand.
    
    cutoff (float, optional): Cutof used for the crosslink distance restraint.
    Defaults to 28.0 A (typical value for DSS crosslinks.)
    """
    
    # read histogram files
    hist_data = OrderedDict()
    for l, fn in histogram_files.items():
        this_data = np.loadtxt(fn)
        hist_data[l] = (this_data[:,0], this_data[:,1])
    
    # plot histograms
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1,1,1)
    
    for k, v in hist_data.items():
        x, y = v
        ax.plot(v[0], v[1], lw=2, color=ligand_colors[k],
                drawstyle="steps", label=k)
    
    ax.axvline(cutoff, ls="--", lw=1.5, color="black")
        
    ax.legend(loc="best", prop={"size": 12})
    ax.set_xlabel("distance " + r"$(\AA)$", fontsize=15)
    ax.set_ylabel("count", fontsize=15)
    
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close()
    

def plot_escape_mutation_distances(datafile, outfile, 
                                   receptor, ligands,
                                   ligand_colors, cutoff=8.0):
    """
    Plot avg. values of closest approach distances between mutant residues
    of receptor and ligands (nanobody CDR3 loops).
    
    Args:
    datafile (str): Escape mutation data files are CSV files with four columns:
    <mutant_residue_key>, <ligand_name>, <ligand_residue_ranges>.

    The mutant residue key is an alphanumeric string with the first character
    as the FASTA letter for that residue, while the remaining characters make up the (int) residue id.
    
    The ligand residue ranges is a string of the form
    "[(b1, e1), (b2, e2), ...]" where (b_i, e_i) is the beginning and end 
    residue id of the ith segment from the ligand (nanobody) which is proximal
    to the receptor mutant residue.
    
    outfile (str): Output file name (.png preferred).
    
    receptor (str): Receptor molecule name.
    
    ligands (str): List of ligand molecule names.
    
    ligand_colors (str): Line-color to use for each ligand.
    
    cutoff (float, optional): Cutoff used for the escape mutation distance
    restraint. Defaults to 8.0 A.
    """
    
    # read escape mutation data files
    mutant_residues, d_mean, d_std, d_centroid, colors = [], [], [], [], []
    df = pd.read_csv(datafile)
    for i in range(len(df)):
        this_df = df.iloc[i]
        mutant_residues.append(this_df["escape_mutant_residue"])        
        d_mean.append(this_df["distance_mean"])
        d_std.append(this_df["distance_std"])
        d_centroid.append(this_df["distance_centroid"])
        colors.append(ligand_colors[this_df["ligand"]])
        
    # plot the mean distances
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1,1,1)
    
    bar_width = 0.3
    idx = np.arange(len(df))
    ax.bar(idx, d_centroid, bar_width, 
           color=colors, edgecolor="black", hatch="//")
    
    ax.bar(idx+bar_width, d_mean, bar_width,
           yerr=d_std, capsize=3,
           color=colors, edgecolor="black", ecolor="black")
    
    ax.axhline(cutoff, ls="--", lw=1.5, color="black")
    
    xlabels = ["%s: %s" % (receptor, m) for m in mutant_residues]
    ax.set_xticks(idx+bar_width/2)
    ax.set_xticklabels(xlabels, rotation=60)
    ax.set_ylabel("closest approach distance " + r"$(\AA)$", fontsize=12)
    
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close()


def plot_epitope_histogram(histogram_file, outfile, 
                           receptor_residues, ligand_colors):
    """
    Plot histograms of epitope residues on the receptor. These are 1D 
    histograms as a function of the receptor residue id.
    
    Args:
    histogram_file (str): ASCII (.txt) file containing the histogram
    in two columns (residue id, epitope probability)
    
    outfile (str): Output file name (.png preferred).
    
    receptor_residues (list): Sorted list of ids for all receptor residues
    that are in the histogram file.
    
    ligand_colors (str): Line-color to use for each ligand.
    """
    # read data
    df = pd.read_csv(histogram_file)
    ligands = list(df.keys())[1:]
    
    nr = len(df)
    epitopes = OrderedDict()
    for i in range(len(df)):
        this_df = df.iloc[i]
        for l in ligands:
            if l not in epitopes: epitopes[l] = []
            epitopes[l].append(this_df[l])
    
    # plot histograms
    fig = plt.figure(figsize=(25,1.5*len(ligands)))
    axs = []
    for i, l in enumerate(ligands):
        ax = fig.add_subplot(len(ligands), 1, i+1)
        color = ligand_colors[l]
        colormap = LinearSegmentedColormap.from_list("hotcold", N=nr,
                                             colors=["white", color])
        e = np.array(epitopes[l]).reshape(1, -1)
        im = ax.pcolormesh(e, cmap=colormap)
        
        ax.plot(epitopes[l], ls="-", lw=2, color="black", drawstyle="steps")
        
        for i in range(e.shape[-1]):
            ax.axvline(i, ls="-", lw=1)
        
        for i in range(1, len(receptor_residues)):
            r = [j for jj in receptor_residues[:i] for j in jj]
            y = len(r)
            ax.axvline(y, ls="-", color="black", lw=4)

        ax.set_xticks([]) ; ax.set_xticklabels([])
        ax.set_yticks([]) ; ax.set_yticklabels([])
        
        ax.set_ylim([0, 1.1])
        ax.set_ylabel(l, fontsize=15)
        
        axs.append(ax)
    
    fig.subplots_adjust(hspace=0.01)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()

