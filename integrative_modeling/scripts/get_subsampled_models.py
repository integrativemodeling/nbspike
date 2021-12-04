"""
Extract multiple subsamples of equilibrated models  produced from structural
sampling. Also, check correlations between the different types of scores.

Author: Tanmoy Sanyal, Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import glob
import json
import argparse
import subprocess
import numpy as np
import pandas as pd

from analysis_trajectories import AnalysisTrajectories
from nblib import utils

DIR_HEAD = "run_"
BFRAC = 0.01
NSKIP = 10
NUM_SUBSAMPLES = 2


def _refresh_HDBSCAN_output(input_dir, output_dir=None, dummy_run=True):
    # if this was an actual score clustering?
    if not dummy_run:
        # copy HDBSCAN output to a separate directory
        fnlist = ["plot_clustering_scores.png", 
                  "summary_hdbscan_clustering.dat"]
        for fn_ in fnlist:
            fn = os.path.join(input_dir, fn_)
            if os.path.isfile(fn):
                new_fn = os.path.join(output_dir, fn_)
                os.system("cp %s %s" % (fn, new_fn))
        
    # remove all clustering related files
    del_fn_list = glob.glob(os.path.join(input_dir, "*cluster*"))
    for fn in del_fn_list:
        if os.path.isfile(fn):
            os.remove(fn)


# take user input
parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-nb", "--nanobody", help="Name of nanobody.")

parser.add_argument("-d", "--datadir", default="data",
        help="Directory containing all data for modeling.")

parser.add_argument("-m", "--modeling_dir", default=".",
        help="Modeling directory containing independent modeling runs")

parser.add_argument("-np", "--nproc", type=int, default=1,
        help="number of processors for parallel run")

args = parser.parse_args()
nanobody = args.nanobody
datadir = os.path.abspath(args.datadir)
modeling_dir = os.path.abspath(args.modeling_dir)
nproc = args.nproc

# config
config_fn = os.path.join(datadir, "config", nanobody + ".json")
with open(config_fn, "r") as of:
    config = json.load(of)
HAS_XL = config.get("HAS_XL", True)
HAS_ESCAPE_MUTANT = config.get("HAS_ESCAPE_MUTANT", True)

# input dirs
traj_dirs = glob.glob(os.path.join(modeling_dir, DIR_HEAD + "*", "output"))

# output dirs
outdir = os.path.join(modeling_dir, "subsampled_models")
scores_dir = os.path.join(outdir, "score_statistics")
scores_correlation_dir = os.path.join(outdir, "score_correlation")
os.makedirs(scores_dir, exist_ok=True)
os.makedirs(scores_correlation_dir, exist_ok=True)

# get receptor and ligand names
topology_fn = os.path.join(datadir, "topology", nanobody+".txt")
th = utils.TopologyHandler(topology_fn,
                           pdb_dir=os.path.join(datadir, "pdb"),
                           fasta_dir=datadir)
assert len(th.ligands) == 1

receptor = th.receptor
ligand = th.ligands[0]


#------------------------------------------
# READ STAT FILES AND CLUSTER SCORING TERMS
# -----------------------------------------
# fields for score based clustering
fields = []

# init analyzer
AT = AnalysisTrajectories(out_dirs=traj_dirs, 
                          dir_name=DIR_HEAD,
                          analysis_dir=scores_dir,
                          burn_in_fraction=BFRAC,
                          nproc=nproc,
                          nskip=NSKIP,
                          detect_equilibration=True,
                          plot_fmt='png')

# excluded volume restraints
AT.set_analyze_Excluded_volume_restraint()
fields.append("EV_sum")

# crosslink distance restraints
if HAS_XL:
    label = utils.get_XL_restraint_label(receptor, ligand)
    AT.set_analyze_score_only_restraint(
        "CrosslinkDistanceScore_" + label, "XLS_%s" % label,
        do_sum=False)
    fields.append("XLS_%s_sum" % label)

# escape mutation distance restraints
if HAS_ESCAPE_MUTANT:
    AT.set_analyze_score_only_restraint("EscapeMutationDistanceScore", "EMDS")
    fields.append("EMDS_sum")

# epitope restraint
AT.set_analyze_score_only_restraint("EpitopeScore", "ES") 
fields.append("ES_sum")

# read stat files (this is slow)
AT.read_stat_files()
AT.write_models_info()

# do score based HDBSCAN clustering
# Note: XL_<receptor>_<ligand> score fields have to be appended by a "_sum"
# suffix to prevent PMI_Analysis from complaining.
AT.hdbscan_clustering(fields, min_cluster_size=500)

# refresh 
_refresh_HDBSCAN_output(input_dir=scores_dir, 
                        output_dir=scores_correlation_dir,
                        dummy_run=False)


# ------------------------------------------------------
# GET A SUBSAMPLE OF MODELS FROM THE EQUILIBRATED MODELS
# ------------------------------------------------------
for i in range(NUM_SUBSAMPLES):
    print("\nSUBSAMPLING RUN %d..." % i)
    print("======================")

    # do a dummy clustering by using a very large min. cluster size
    # such that only cluster containing a subsample of models is produced
    AT.hdbscan_clustering(fields, min_cluster_size=int(1e10))
    
    # get dataframes
    score_fn_A = "selected_models_A_cluster-1_detailed_random.csv"
    score_fn_B = "selected_models_B_cluster-1_detailed_random.csv"
    HA = AT.get_models_to_extract(os.path.join(scores_dir, score_fn_A))
    HB = AT.get_models_to_extract(os.path.join(scores_dir, score_fn_B))
    
    # extract subsampled models and scores for partition A
    AT.do_extract_models_single_rmf(gsms_info=HA,
                    out_rmf_name="models_A.rmf3",
                    traj_dir=indir,
                    analysis_dir=outdir,
                    scores_prefix="scores_A")
    
    # extract subsampled models and scores for partition B
    AT.do_extract_models_single_rmf(gsms_info=HB,
                    out_rmf_name="models_B.rmf3",
                    traj_dir=indir,
                    analysis_dir=outdir,
                    scores_prefix="scores_B")
    
    # combine the two RMF files into a single rmf file
    in_rmf_files = [os.path.join(outdir, "models_%s.rmf3" % h) 
                    for h in ["A", "B"]]
    out_rmf_file = os.path.join(outdir, "models_%d.rmf3" % i)
    subprocess.check_call(["rmf_cat"] + in_rmf_files + [out_rmf_file])
    
    # combine the two score files into a single score file
    in_score_files = [os.path.join(outdir, "scores_%s.txt" % h)
                      for h in ["A", "B"]]
    
    # delete files for individual partitions
    del_fn_list = ["models_A.rmf3", "models_B.rmf3", 
                   "scores_A.txt", "scores_B.txt"]
    for fn_ in del_fn_list:
        fn = os.path.join(outdir, fn_)
        if os.path.isfile(fn): os.remove(fn)
            
    # refresh
    _refresh_HDBSCAN_output(input_dir=scores_dir, dummy_run=True)
    print("\n")
    
