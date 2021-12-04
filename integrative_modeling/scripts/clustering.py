"""
Cluster docked receptor + nanobody complexes based on the 
fraction of common contacts (FCC) metric.

Author: Tanmoy Sanyal, Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import json
import argparse
from nblib import InterfaceCluster

READ_FREQ = 50

# read and parse user input
parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-nb", "--nanobody", help="Name of nanobody.")

parser.add_argument("-r", "--rmf_file",
            help="RMF file containing all good scoring models.")

parser.add_argument("-d", "--datadir",
            help="Directory containing data for integrative modeling.")

parser.add_argument("-o", "--outdir", help="Output directory.")

parser.add_argument("-n", "--num_omp_threads", type=int, default=2,
                    help="Number of OMP threads for parallel execution")

args = parser.parse_args()
nanobody = args.nanobody
rmf_fn = os.path.abspath(args.rmf_file)
datadir = os.path.abspath(args.datadir)
outdir = os.path.abspath(args.outdir)
num_omp_threads = args.num_omp_threads

# config
config_fn = os.path.join(datadir, "config", nanobody + ".json")
with open(config_fn, "r") as of:
    config = json.load(of)
EPITOPE_CUTOFF = config.get("INTERFACE_CLUSTER_EPITOPE_CUTOFF", 8.0)
CLUSTER_CUTOFF = config.get("INTERFACE_CLUSTER_CUTOFF", 0.25)
MIN_CLUSTER_SIZE = config.get("INTERFACE_CLUSTER_MIN_CLUSTER_SIZE", 10)

# init the clustering object
topology_fn = os.path.join(datadir, "topology", nanobody + ".txt")
pdb_dir = os.path.join(datadir, "pdb")
cl = InterfaceCluster(rmf_fn, topology_fn,
                    pdb_dir=pdb_dir,
                    fasta_dir=datadir,
                    read_freq=READ_FREQ,
                    num_omp_threads=num_omp_threads,
                    outdir=outdir)

# calculate inteface distance matrix
# this is an expensive operation and so, is cached after the first run
cl.calculate_interface_distance_matrix(EPITOPE_CUTOFF)

# run the clustering algorithm (Taylor Butina)
cl.cluster(cluster_cutoff=CLUSTER_CUTOFF, 
           min_cluster_size=MIN_CLUSTER_SIZE)

# calculate precison for all clusters
cl.calculate_precision()

