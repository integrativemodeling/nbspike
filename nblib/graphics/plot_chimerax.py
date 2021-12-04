"""
Utility functions for writing UCSF ChimeraX scripts for 3D rendering
of crosslinks and epitopes of docked nanobody models on the SARS-CoV2 
Spike protein and its domains.

Note: This is only compatible with ChimeraX, and will not work for Chimera.
"""

import os
import pandas as pd
import numpy as np
from collections import OrderedDict

from ..utils import RMFHandler
from ..utils import read_XL_data, read_escape_mutant_data

S_PSEUDOBOND_HEADER = """
; radius=0.5
; dashes=0
"""

XL_COLORS = {0: "red", 1: "blue"}

def render_cluster_variability(pdb_file, receptor, ligands, 
                               molinfo, num_receptor_copies, 
                               outfile):
    """
    Write ChimeraX script for rendering top 10 models from a given cluster.

    Args:
    pdb_file (str): PDB file containing 10 best models from a given cluster.
    
    receptor (str): Name of receptor molecule.
    
    ligands (list): List of names of ligand molecules.
    
    molinfo (dict): Dict containing PDB chain and residue information
    about each molecule in the system. This information is produced
    by utils.TopologyHandler
    
    num_receptor_copies (int): Number of copies of the receptor. For the 
    SARS-CoV2 Spike protein, this number is 3.
    
    outfile (str): Output file (must have .cxc extension).
    """
    
    s = "set bgColor transparent\n"
    s += "open %s\n" % os.path.basename(pdb_file)
    s += "hide all\n"
    
    # receptor
    for i in range(num_receptor_copies):
        key = receptor + "." + str(i)
        r_info = molinfo[key]
        s += "# receptor, copy %d\n" % i
        s += "surface #*/%s\n" % r_info["tar_pdb_chain"]
        s += "color #*/%s %s\n" % (r_info["tar_pdb_chain"], r_info["color"])
        s += "\n"
    
    # ligands
    for l in ligands:
        l_info = molinfo[l]
        s += "# ligand %s\n" % l
        s += "cartoon #*/%s\n" % l_info["tar_pdb_chain"]
        s += "color #*/%s %s\n" % (l_info["tar_pdb_chain"], l_info["color"])
        s += "\n"
        
    with open(outfile, "w") as of:
        of.write(s)


def render_crosslink_maps(pdb_file, receptor, ligands,
                          molinfo, num_receptor_copies,
                          centroid_sat, pseudobond_dir, outfile):
    """
    Write ChimeraX script for rendering crosslinks as pseudobonds
    on ribbon representations of molecules.

    Args:
    pdb_file (str): PDB file containing a representative docked complex
    of the receptor and all ligands.
    
    receptor (str): Name of receptor molecule.
    
    ligands (list): List of names of ligand molecules.
    
    molinfo (dict): Dict containing PDB chain and residue information
    about each molecule in the system. This information is produced
    by utils.TopologyHandler
    
    num_receptor_copies (int): Number of copies of the receptor. For the 
    SARS-CoV2 Spike protein, this number is 3.
    
    centroid_sat (dict): Dictionary containing satisfaction info for
    each crosslink dataset, i.e. for each ligand. 
    
    pseudobond_dir (str): Directory into which pseudobond settings files
    will be written for each crosslink dataset. Pseudobonds corresponding
    to satisfied crosslinks will be colored blue and those for violated
    crosslinks will be colored red.
    
    For more details on pseudobond files, see:
    https://www.cgl.ucsf.edu/chimerax/docs/user/pseudobonds.html
    
    outfile (str): Output file (must have .cxc extension).
    """
    
    assert sorted(ligands) == sorted(list(centroid_sat.keys()))
    
    # write pseudobond models
    for l in ligands:
        s_pb = S_PSEUDOBOND_HEADER
        for k, v in centroid_sat[l].items():
            r, r_res, l, l_res = k
            r_chain = molinfo[r]["tar_pdb_chain"]
            l_chain = molinfo[l]["tar_pdb_chain"]
            pb_color = XL_COLORS[int(v)]
            s_pb += "#1/%s:%d@CA #1/%s:%d@CA %s\n" % (r_chain, r_res,
                                                      l_chain, l_res,
                                                      pb_color)
        out_pb_fn = os.path.join(pseudobond_dir, "%s.pb" % l)
        with open(out_pb_fn, "w") as of:
            of.write(s_pb)
    
    # write chimerax script
    s = "set bgColor transparent\n"
    s += "open %s\n" % os.path.basename(pdb_file)
    s += "hide all\n"
    s += "cartoon\n"
    
    # color receptor copies
    for i in range(num_receptor_copies):
        key = receptor + "." + str(i)
        r_info = molinfo[key]
        s += "# receptor, copy %d\n" % i
        s += "color #*/%s %s\n" % (r_info["tar_pdb_chain"],
                                   r_info["color"])
    s += "\n"
    
    # color ligands
    s += "# ligands\n"
    for l in ligands:
        s += "color #1/%s %s\n" % (molinfo[l]["tar_pdb_chain"],
                                   molinfo[l]["color"])
    s += "\n"
    
    # open pseudobond models
    s += "# crosslinks\n"
    for l in ligands:
        s += "open %s.pb\n" % l
    s += "\n"
    
    with open(outfile, "w") as of:
        of.write(s)


def render_epitope_maps(pdb_file, receptor, ligands, molinfo, 
                        num_receptor_copies, centroid_epitopes,
                        outdir):
    """
    Write *separate* ChimeraX scripts for rendering epitopes for each ligand
    on to a surface representation of the receptor.

    Args:
    pdb_file (str): PDB file containing a representative docked complex
    of the receptor and all ligands.
    
    receptor (str): Name of receptor molecule.
    
    ligands (list): List of names of ligand molecules.
    
    molinfo (dict): Dict containing PDB chain and residue information
    about each molecule in the system. This information is produced
    by utils.TopologyHandler
    
    num_receptor_copies (int): Number of copies of the receptor. For the 
    SARS-CoV2 Spike protein, this number is 3.
    
    centroid_epitopes (list): List of (copy number, residue id) for each
    epitope residue on the receptor.
    
    outdir (str): Output directory. Chimerax script for a ligand epitope
    will be named as <ligand_name>_epitope.cxc.
    """
    
    s0 = "set bgColor transparent\n"
    s0 += "open %s\n" % os.path.basename(pdb_file)
    s0 += "hide\n"
    s0 += "\n"
    
    # receptor
    for i in range(num_receptor_copies):
        key = receptor + "." + str(i)
        r_info = molinfo[key]
        s0 += "# receptor, copy %d\n" % i
        s0 += "surface #*/%s\n" % r_info["tar_pdb_chain"]
        s0 += "color #*/%s %s\n" % (r_info["tar_pdb_chain"], r_info["color"])
        s0 += "\n"
    
    for l in ligands:
        s = s0
        # ligand
        s += "# ligand\n"
        s += "cartoon #*/%s\n" % (molinfo[l]["tar_pdb_chain"])
        s += "color #*/%s %s\n" % (molinfo[l]["tar_pdb_chain"],
                                   molinfo[l]["color"])
        s += "\n"
        
        # epitope
        s += "# epitope\n"
        residue_dict = OrderedDict()
        for (c, r) in centroid_epitopes[l]:
            if c not in residue_dict:
                residue_dict[c] = []
            residue_dict[c].append(r)
        
        for k, v in residue_dict.items():
            res_str = ",".join([str(i) for i in sorted(v)])
            chain = molinfo[receptor + "." + str(k)]["tar_pdb_chain"]
            color = molinfo[l]["color"]
            s += "color #*/%s:%s %s target s\n" % (chain, res_str, color)
        s += "\n"
        
        outfn = os.path.join(outdir, "%s_epitope.cxc" % l)
        with open(outfn, "w") as of:
            of.write(s)


def render_binary_docking_epitope(pdb_file, receptor, ligand, molinfo,
                        num_receptor_copies, outfile, 
                        target_receptor_copies=[0],
                        xl_file=None, escape_mutant_file=None):
    
    s = "set bgColor transparent\n"
    s += "open %s\n" % os.path.basename(pdb_file)
    s += "hide\n"
    s += "\n"

    # receptor
    for i in range(num_receptor_copies):
        key = receptor + "." + str(i)
        r_info = molinfo[key]
        s += "# receptor, copy %d\n" % i
        s += "surface #1/%s\n" % r_info["tar_pdb_chain"]
        s += "color #1/%s %s\n" % (r_info["tar_pdb_chain"], "silver")
    s += "\n"

    # ligand
    l_info = molinfo[ligand]
    s += "# ligand\n"
    s += "cartoon #1/%s\n" % (l_info["tar_pdb_chain"])
    s += "color #1/%s pink\n" % l_info["tar_pdb_chain"]
    s += "\n"

    # epitope
    s += "# epitope\n"
    s += "select zone #1/%s 6 #1/%s ; color sel pink; ~sel" % \
        (l_info["tar_pdb_chain"], r_info["tar_pdb_chain"])
    s += "\n"

    # crosslinks
    if xl_file is not None:
        s += "# crosslinks\n"
        xl_pairs = read_XL_data(xl_file)
        xl_residues = sorted({r1 for (p1,r1,p2,r2) in xl_pairs})
        xl_str = ",".join([str(x) for x in xl_residues])
        for i in target_receptor_copies:
            key = receptor + "." + str(i)
            r_info = molinfo[key]
            s += "color #1/%s:%s yellow target s\n" % (r_info["tar_pdb_chain"],
                                                 xl_str)
        s += "\n"

    # escape mutations
    if escape_mutant_file is not None:
        s += "# escape mutants\n"
        parsed_data = read_escape_mutant_data(
        escape_mutant_file)

        mutant_residues = set()
        ligand_residues = set()
        for rn, rr, lrs in parsed_data:
            mutant_residues |= set([rr])
            ligand_residues |= set(lrs)
        mutant_residues = sorted(mutant_residues)
        ligand_residues = sorted(ligand_residues)
        mutant_str = ",".join([str(x) for x in mutant_residues]) 
        ligand_str = ",".join([str(x) for x in ligand_residues])
        for i in target_receptor_copies:
            key = receptor + "." + str(i)
            r_info = molinfo[key]
            s += "color #1/%s:%s red\n" % (r_info["tar_pdb_chain"], mutant_str)
        s += "color #1/%s:%s navy \n" % (l_info["tar_pdb_chain"], ligand_str)

    with open(outfile, "w") as of:
        of.write(s)
        


        