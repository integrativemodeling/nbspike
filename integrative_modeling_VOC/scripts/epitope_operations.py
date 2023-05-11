import os
import json
import numpy as np
from scipy.spatial import cKDTree

import argparse
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import get_surface
from Bio.SeqUtils import seq1

_MUTATION_FILE = os.path.join(os.path.dirname(__file__), "data/VOC_spike_mutations.json")
_SURFACE_CUTOFF = 8 # A
_EPITOPE_CUTOFF = 6 # A


def _get_receptor_and_ligand_residues(pdb_file):
    model = PDBParser(QUIET=True).get_structure("x", pdb_file)[0]
    receptor = model["0"]
    nb = model["A"]
    return receptor, nb

def _get_receptor_proximal_residues(receptor_residues, ref_coords, cutoff):
    atom2res = {}
    coords = []
    count = 0
    for i, r in enumerate(receptor_residues):
        for a in r.get_atoms():
            coords.append(a.coord)
            atom2res[count] = i
            count += 1
    coords = np.array(coords)
    
    tree = cKDTree(np.array(coords))
    ref_tree = cKDTree(np.array(ref_coords))
    
    indices = tree.query_ball_tree(ref_tree, r=cutoff)
    res_indices = {atom2res[ii] for ii, i in enumerate(indices) if i}
    return [receptor_residues[i] for i in res_indices]


def get_epitope(pdb_file):
    receptor, nb = _get_receptor_and_ligand_residues(pdb_file)
    surface_point_coords = get_surface(receptor)
    receptor_surface_residues = _get_receptor_proximal_residues(
        receptor_residues=list(receptor.get_residues()),
        ref_coords=surface_point_coords,
        cutoff=_SURFACE_CUTOFF)
    
    nb_coords = []
    for r in nb.get_residues():
        if r.id[1] in range(1, 7): continue
        nb_coords.extend([a.coord for a in r.get_atoms()])
    nb_coords = np.array(nb_coords)
    
    epitope = _get_receptor_proximal_residues(
                receptor_residues=receptor_surface_residues,
                ref_coords=nb_coords,
                cutoff=_EPITOPE_CUTOFF)

    epitope_coords = np.array([a.coord for r in epitope for a in r.get_atoms()])
    epitope_center_coord = np.mean(epitope_coords,axis=0)
    
    epitope_centers = _get_receptor_proximal_residues(
        receptor_residues=epitope,
        ref_coords=epitope_center_coord.reshape((1,3)),
        cutoff=_SURFACE_CUTOFF)
    
    epitope_center_str = seq1(epitope_centers[0].resname).upper() + \
                         str(epitope_centers[0].id[1])
    
    return epitope, epitope_center_str


def get_consensus_epitope(pdb_files):
    epitope_resids = []
    for f in pdb_files:
        e, _ = get_epitope(f)
        resids = {r.id[1] for r in e}
        epitope_resids.append(resids)
    return set.union(*epitope_resids)


def get_center_between_two_epitopes(pdb_files_1, pdb_files_2, 
                                    compare="and"):
    e1 = get_consensus_epitope(pdb_files_1)
    e2 = get_consensus_epitope(pdb_files_2)
    
    if compare == "and":
        e = set.intersection(e1, e2)
    elif compare == "or":
        e = set.union(e1, e2)
    receptor, _ = _get_receptor_and_ligand_residues(pdb_files_1[0])
    all_residues = list(receptor.get_residues())
    e_residues = [r for r in all_residues if r.id[1] in e]
    e_coords = np.array([a.coord for r in e_residues for a in r.get_atoms()])
    e_center_coord = np.mean(e_coords, axis=0)
    e_center = _get_receptor_proximal_residues(
        receptor_residues=e_residues,
        ref_coords=e_center_coord.reshape((1,3)),
        cutoff=_SURFACE_CUTOFF)
    return e, e_center


def get_proximal_mutations(pdb_file, variant):
    receptor, nb = _get_receptor_and_ligand_residues(pdb_file)
    e, _ = get_epitope(pdb_file)
    e_coords = np.array([a.coord for r in e for a in r.get_atoms()]) 
    
    with open(_MUTATION_FILE, "r") as of:
        mutation_resids = [int(m[1:-1]) for m in json.load(of)[variant]]
    mutation_residues = [r for r in receptor if r.id[1] in mutation_resids]
    
    proximal_residues = _get_receptor_proximal_residues(
        receptor_residues=mutation_residues,
        ref_coords=e_coords,
        cutoff=6.0)
    
    return proximal_residues


# user input
parser = argparse.ArgumentParser(
    description="Find center of an epitope of center between two epitopes")

parser.add_argument("-p", "--pdb_file",  
    help="Co-complex of Nb on Wuhan receptor.")

parser.add_argument("-ps", "--pdb_files", nargs="+",
    help="Multiple PDB files from the same Wuhan epitope group.")

parser.add_argument("-ps1", "--pdb_files_1", nargs="+",
    help="Multiple PDB files from the same Wuhan epitope group.")

parser.add_argument("-ps2", "--pdb_files_2", nargs="+",
    help="Multiple PDB files from the same Wuhan epitope group.")

parser.add_argument("-e", "--epitope", action="store_true",
    help="True to calculate epitope from pdb file supplied with argument '-p'")

parser.add_argument("-c", "--consensus_epitope", action="store_true",
    help="True to calculate the consensus epitope from pdb files supplied with argument '-ps'")

parser.add_argument("-b", "--center_between_epitopes", action="store_true",
    help="True to calculate the center between epitopes from pdb file groups supplied with arguments '-ps1' and '-ps2'")

parser.add_argument("-m", "--proximal_mutations", action="store_true",
    help="True to calculate mutations proximal to an epitope")

parser.add_argument("-v", "--variant", choices=["delta", "omicron"],
    help="Variant used for analysing proximal mutations")

args = parser.parse_args()


# epitope
if args.epitope:
    e, e_center_str = get_epitope(args.pdb_file)
    reslist = sorted([r.id[1] for r in e])
    s = ", ".join([str(x) for x in reslist])
    print("Epitope residues:", s)
    print("Center of the epitope:", e_center_str)

# consensus epitope
if args.consensus_epitope:
    residues = get_consensus_epitope(args.pdb_files)
    s = ",".join([str(r) for r in residues])
    print("Consensus epitope residues:", s)

# center between epitopes
if args.center_between_epitopes:
    e, e_center = get_center_between_two_epitopes(args.pdb_files_1,
                                                  args.pdb_files_2)
    s = ",".join([seq1(r.resname).upper() + str(r.id[1]) for r in e_center])
    print("Union epitope residues:", ",".join([str(i) for i in e]))
    print("Center between the epitope groups: ", s)

# proximal mutations
if args.proximal_mutations:
    residues = get_proximal_mutations(args.pdb_file, args.variant)    
    reslist = sorted([r.id[1] for r in residues])
    s = ", ".join([str(x) for x in reslist])
    print("Proximal mutations:", s)    
