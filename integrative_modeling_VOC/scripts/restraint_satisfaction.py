"""
Check input information satisfaction by clustered models.

Author: Tanmoy Sanyal, Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import json
import argparse
import shutil
import numpy as np
from collections import OrderedDict

from nblib import restraint_satisfaction as rs

VARIANTS = ["wuhan", "delta", "omicron", "ba4"]

# helper function to calculate restraint satisfaction for each cluster
def _check_cluster(variant, nanobody, rmf_fn, 
                   datadir, cluster_dir, outdir, 
                   n_threads):
    
    # topology file
    topology_fn = os.path.join(datadir, "topology", variant, nanobody + ".txt")

    # config
    config_fn = os.path.join(datadir, "config", variant, nanobody+".json")
    with open(config_fn, "r") as of:
        config = json.load(of)

    # centroid pdb
    centroid_pdb_fn = os.path.join(cluster_dir, "cluster_center_model.pdb")
    shutil.copyfile(centroid_pdb_fn,
                    os.path.join(outdir, os.path.basename(centroid_pdb_fn)))
    
    # frames
    cluster_frames_fn = os.path.join(cluster_dir, "frames.txt")
    frames = list(np.loadtxt(cluster_frames_fn, dtype=int))
    
    # escape mutation distance restraint satisfaction
    has_escape_mutant = config.get("HAS_ESCAPE_MUTANT", True)
    escape_mutant_fn = {nanobody: None}
    if has_escape_mutant:
        emdr_cutoff = config.get("EMDR_CUTOFF", 8.0) + \
                      config.get("DELTA_EMDR_CUTOFF", 1.0)
        escape_mutant_fn[nanobody] = os.path.join(datadir, "escape_mutant",
                                                variant,
                                                nanobody + ".csv")
        
        rs_ems = rs.EscapeMutationSatisfaction(topology_file=topology_fn,
                                    rmf_file=rmf_fn,
                                    centroid_pdb_file=centroid_pdb_fn,
                                    pdb_dir=os.path.join(datadir, "pdb"),
                                    fasta_dir=datadir,
                                    centroid_frame=frames[0],
                                    frames=frames,
                                    outdir=outdir,
                                    num_omp_threads=n_threads)
        rs_ems.add_data(escape_mutant_fn, cutoff=emdr_cutoff)
        rs_ems.run()
    
    # epitope overlap
    epitope_cutoff = config.get("INTERFACE_CLUSTER_EPITOPE_CUTOFF", 8.0)
    rs_ep = rs.Epitope(topology_file=topology_fn,
                    rmf_file=rmf_fn,
                    centroid_pdb_file=centroid_pdb_fn,
                    centroid_frame=frames[0],
                    frames=frames,
                    outdir=outdir,
                    num_omp_threads=n_threads)
    rs_ep.setup(cutoff=epitope_cutoff)
    rs_ep.run()

    target_receptor_copies = config.get("RECEPTOR_COPIES", [0])
    rs_ep.render_binary_docking_epitope(XL_file=None,
                            escape_mutant_file=escape_mutant_fn[nanobody],
                            target_receptor_copies=target_receptor_copies)



#### MAIN ####

# read and parse user input
parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-v", "--variant", 
                    help="Name of SARS-CoV2 variant [delta or omicron].")

parser.add_argument("-nb", "--nanobody", help="Name of nanobody.")

parser.add_argument("-r", "--rmf_file",
            help="RMF file containing all subsampled models.")

parser.add_argument("-d", "--datadir",
            help="Directory containing data for integrative modeling.")

parser.add_argument("-c", "--cluster_dir", 
            help="Directory containing all the clusters")

parser.add_argument("-o", "--outdir", help="Output directory")

parser.add_argument("-n", "--num_omp_threads", type=int, default=2,
            help="Number of OMP threads for parallel execution")

args = parser.parse_args()
variant = args.variant
nanobody = args.nanobody
rmf_fn = os.path.abspath(args.rmf_file)
datadir = os.path.abspath(args.datadir)
cluster_dir = os.path.abspath(args.cluster_dir)
outdir = os.path.abspath(args.outdir)
num_omp_threads = args.num_omp_threads

assert variant in VARIANTS

# get number of clusters
cluster_size_fn = os.path.join(cluster_dir, "cluster_size.txt")
if not os.path.isfile(cluster_size_fn):
    exit()
cluster_sizes = np.loadtxt(cluster_size_fn)
n_clusters = len(cluster_sizes.reshape(-1,2))

for i in range(n_clusters):
    this_cluster_dir = os.path.join(cluster_dir, "cluster.%d" % i)
    if not os.path.isdir(this_cluster_dir):
        print("WARNING: Cluster %d not found. Skipping ahead." % i)
        continue
    
    print("\n----------------CLUSTER %d----------------------" % i)
    this_outdir = os.path.join(outdir, "cluster.%d" % i)
    os.makedirs(this_outdir, exist_ok=True)
    _check_cluster(variant, nanobody, rmf_fn, datadir, 
                  this_cluster_dir, this_outdir, num_omp_threads)
