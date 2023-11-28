"""
Integrative modeling of the binary docking of 
nanobodies to the SARS-CoV2 Spike protein domains.

Author: Tanmoy Sanyal, Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import json
import argparse

import IMP, IMP.atom
import IMP.pmi.topology
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.macros
import IMP.pmi.tools

from nblib import restraints, utils

# variants covered in this repository
VARIANTS = ["wuhan", "delta", "omicron", "ba4", "xbb", "bq"]

import numpy as np
np.random.seed = int(10000 * np.random.random())

# rigid body movement parameters
MAX_RB_TRANS = 1.00
MAX_RB_ROT = 0.05
MAX_SRB_TRANS = 1.00
MAX_SRB_ROT = 0.05
MAX_BEAD_TRANS = 2.00  # flexible region / bead movement

# randomize initial config parameters
SHUFFLE_ITER = 100
INIT_MAX_TRANS = 500.0

# replica exchange parameters
MC_TEMP = 1.0
MIN_REX_TEMP = 1.0
MAX_REX_TEMP = 2.5

# simulated annealing parameters
MIN_SA_TEMP = 1.0
MAX_SA_TEMP = 2.5

# sampling iterations
MC_FRAMES_1 = 10000
MC_FRAMES_2 = 10000
MC_STEPS_PER_FRAME_1 = 20
MC_STEPS_PER_FRAME_2 = 40


# read and parse user input
parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-v", "--variant", 
                    help="Name of SARS-CoV2 variant [delta or omicron].")

parser.add_argument("-nb", "--nanobody", help="Name of nanobody.")

parser.add_argument("-d", "--datadir", default="data",
                    help="data directory")

parser.add_argument("-t", "--test", action="store_true",
                    help="true to test with very limited sampling")

args = parser.parse_args()
variant = args.variant
nanobody = args.nanobody
datadir = os.path.abspath(args.datadir)
is_test = args.test

assert variant in VARIANTS

# test mode sampling iterations
if is_test:
    MC_FRAMES_1 = 100
    MC_FRAMES_2 = 500


# ----------------
# RESTRAINT CONFIG
# ----------------
config_fn = os.path.join(datadir, "config", variant, nanobody + ".json")
with open(config_fn, "r") as of:
    config = json.load(of)

# receptor copies
RECEPTOR_COPIES = config.get("RECEPTOR_COPIES", [])

# exluded volume restraint parameters
EV_KAPPA = config.get("EV_KAPPA", 1.0)
EV_WEIGHT = config.get("EV_WEIGHT", 1.0)

# crosslink distance restraint parameters
HAS_XL = config.get("HAS_XL", True)
XL_CUTOFF = config.get("XL_CUTOFF", 30.0)
XL_KAPPA = config.get("XL_KAPPA", 1.0)
XL_WEIGHT = config.get("XL_WEIGHT", 1.0)

# escape mutation distance restraint parameters
HAS_ESCAPE_MUTANT = config.get("HAS_ESCAPE_MUTANT", True)
EMDR_CUTOFF = config.get("EMDR_CUTOFF", 8.0)
EMDR_KAPPA = config.get("EMDR_KAPPA", 1.0)
EMDR_WEIGHT = config.get("EMDR_WEIGHT", 1.0)

# epitope restraint parameters
HAS_EPITOPE_RESTRAINT = config.get("HAS_EPITOPE_RESTRAINT", True)
ER_CUTOFF = config.get("ER_CUTOFF", 10.0)
ER_EPITOPE_CUTOFF = config.get("ER_EPITOPE_CUTOFF", 15.0)
ER_KAPPA = config.get("ER_KAPPA", 1.0)
ER_WEIGHT = config.get("ER_WEIGHT", 1.0)


# --------------
# REPRESENTATION
# --------------
topology_fn = os.path.join(datadir, "topology", variant, nanobody + ".txt")
th = utils.TopologyHandler(topology_fn,
                           pdb_dir=os.path.join(datadir, "pdb"),
                           fasta_dir=datadir)

assert len(th.ligands) == 1

receptor = th.receptor
ligand = th.ligands[0]
molinfo = th.molinfo
receptor_offset = th.receptor_offset

print("\n--------------------------------------------------------")
print("BINARY DOCKING OF RECEPTOR %s WITH NANOBODY %s" % (receptor, ligand))
print("--------------------------------------------------------")

m = IMP.Model()
bs = IMP.pmi.macros.BuildSystem(m, resolutions=[1])
bs.add_state(th.PMI_topology)
root_hier, dof = bs.execute_macro(max_rb_trans=MAX_RB_TRANS,
                                  max_rb_rot=MAX_RB_ROT,
                                  max_bead_trans=MAX_BEAD_TRANS,
                                  max_srb_trans=MAX_SRB_TRANS,
                                  max_srb_rot=MAX_SRB_ROT)

restraint_list = []


# -----------------------------------------------
# RECEPTOR-LIGAND EXCLUDED VOLUME RESTRAINTS
# -----------------------------------------------
evr_label = utils.get_excluded_volume_restraint_label(receptor, ligand)
if not RECEPTOR_COPIES:
    evr_receptor = bs.get_molecule(receptor)
else:
    evr_receptor = [bs.get_molecule(receptor, copy_index=i)
                    for i in RECEPTOR_COPIES]
evr_ligand = bs.get_molecule(ligand)

receptor_copy = 0 if not RECEPTOR_COPIES else RECEPTOR_COPIES
evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
    included_objects=evr_receptor, other_objects=evr_ligand,
    resolution=1)
evr.set_weight(EV_WEIGHT)
restraint_list.append(evr)


# -----------------------------------------------
# CONNECTIVITY RESTRAINT FOR THE RECEPTOR
# (to handle missing residues)
# (somewhat dummy restraint so don't monitor this)
# ------------------------------------------------
cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(
    bs.get_molecule(receptor))
cr.set_weight(1.0)
restraint_list.append(cr)


# ------------------
# CROSSLINK RESTRAINT
# -------------------
if HAS_XL:
    label = utils.get_XL_restraint_label(receptor, ligand)
    xl_fn = os.path.join(datadir, "xl_wuhan", ligand + ".csv") # only wuhan
    xl_pairs = utils.read_XL_data(xl_fn, r_offset=receptor_offset)
    receptor_copy = 0 if not RECEPTOR_COPIES else RECEPTOR_COPIES[0]
    xlr = restraints.CrosslinkDistanceRestraint(
        receptor_copy=receptor_copy,
        root_hier=root_hier, XL_pairs=xl_pairs,
        cutoff=XL_CUTOFF, kappa=XL_KAPPA,
        resolution=1, label=label)
    xlr.set_weight(XL_WEIGHT)
    restraint_list.append(xlr)


# -------------------------------------
# ESCAPE MUATION DISTANCE RESTRAINT/(S)
# -------------------------------------
mutant_residues = set()
ligand_CDR_residues = set()

if HAS_ESCAPE_MUTANT:
    escape_mutant_fn = os.path.join(datadir, "escape_mutant", variant, 
                                    ligand + ".csv")
    parsed_escape_mutant_data = utils.read_escape_mutant_data(
            filename=escape_mutant_fn, r_offset=receptor_offset)

    if not isinstance(EMDR_WEIGHT, list):
        EMDR_WEIGHT = [EMDR_WEIGHT] * len(parsed_escape_mutant_data)

    for i, (rn, rr, lrs) in enumerate(parsed_escape_mutant_data):
        if lrs is not None:
            ligand_CDR_residues |= set(lrs)

        if not (rn is None or rr is None) and HAS_ESCAPE_MUTANT:
            mutant_residues |= {rr}
            label = utils.get_escape_mutation_distance_restraint_label(
                receptor, ligand, rr)
    
            emdr = restraints.EscapeMutationDistanceRestraint(
                        receptor_copies=RECEPTOR_COPIES,
                        root_hier=root_hier, resolution=1, 
                        receptor_name=receptor, ligand_name=ligand,
                        receptor_residue=rr, ligand_residues=lrs,
                        kappa=EMDR_KAPPA, cutoff=EMDR_CUTOFF, label=label)
            emdr.set_weight(EMDR_WEIGHT[i])
            restraint_list.append(emdr)


# -----------------
# EPITOPE RESTRAINT
# -----------------
if HAS_EPITOPE_RESTRAINT:
    label = utils.get_epitope_restraint_label(receptor, ligand)
    er = restraints.EpitopeRestraint(receptor_copies=RECEPTOR_COPIES,
                            root_hier=root_hier, resolution=1,
                            receptor_name=receptor, ligand_name=ligand,
                            ligand_residues=list(ligand_CDR_residues),
                            epitope_center_residues=list(mutant_residues), 
                            epitope_cutoff=ER_EPITOPE_CUTOFF,
                            cutoff=ER_CUTOFF, kappa=ER_KAPPA,
                            label=label)
    
    er.set_weight(ER_WEIGHT)
    restraint_list.append(er)


# --------
# SAMPLING
# --------
# remove the rigid part of receptor from the degrees of freedom, i.e. this will
# be unaffected by monte-carlo movers.
dof.disable_movers(objects=bs.get_molecule(receptor).get_atomic_residues())

# shuffle only ligand particles to randomize the system i.e. don't re-initialize
# the receptor position between independent runs
IMP.pmi.tools.shuffle_configuration(bs.get_molecule(ligand),
                                    max_translation=INIT_MAX_TRANS,
                                    niterations=SHUFFLE_ITER)


# add all restraints to the model
for r in restraint_list:
    r.add_to_model()

# run replica exchange Monte-Carlo for the first time
print("\nWARM-UP RUNS")
rex1 = IMP.pmi.macros.ReplicaExchange(m, root_hier,
                        monte_carlo_sample_objects=dof.get_movers(),
                        global_output_directory="./output_warmup",
                        output_objects=restraint_list,
                        write_initial_rmf=True,

                        monte_carlo_steps=MC_STEPS_PER_FRAME_1,
                        number_of_frames=MC_FRAMES_1,
                        number_of_best_scoring_models=0,

                        simulated_annealing=True,
                        simulated_annealing_minimum_temperature=MIN_SA_TEMP,
                        simulated_annealing_maximum_temperature=MAX_SA_TEMP,

                        monte_carlo_temperature=MC_TEMP,
                        replica_exchange_minimum_temperature=MIN_REX_TEMP,
                        replica_exchange_maximum_temperature=MAX_REX_TEMP)
rex1.execute_macro()

# run replica exchange Monte-Carlo again
print("\nPRODUCTION RUNS")
rex2 = IMP.pmi.macros.ReplicaExchange(m, root_hier,
                        monte_carlo_sample_objects=dof.get_movers(),
                        global_output_directory="output",
                        output_objects=restraint_list,
                        write_initial_rmf=True,
                        
                        monte_carlo_steps=MC_STEPS_PER_FRAME_2,
                        number_of_frames=MC_FRAMES_2,
                        number_of_best_scoring_models=0,
                        
                        monte_carlo_temperature=MC_TEMP,
                        replica_exchange_minimum_temperature=MIN_REX_TEMP,
                        replica_exchange_maximum_temperature=MAX_REX_TEMP,

                        replica_exchange_object=rex1.replica_exchange_object)
rex2.execute_macro()
