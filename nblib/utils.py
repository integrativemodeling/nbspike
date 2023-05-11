"""
Miscellaneous utility functions / classes for modeling, analysis and validation
of nanobody binding to the SARS-CoV2 spike protein. Nanobodies are referred to
as ligands.
"""

import string
import numpy as np
import pandas as pd
from collections import OrderedDict

import IMP
import IMP.algebra
import IMP.atom
import IMP.core
import IMP.rmf
import IMP.pmi.topology
import RMF 


_CONNOLLY_SURFACE_SAMPLING_DENSITY = 0.15


def get_excluded_volume_restraint_label(receptor, ligand):
    """
    Get label for inter-receptor-ligand excluded volume restraint.
    
    Args:
    receptor (str): Name of receptor molecule.
    
    ligand (str): Name of ligand molecule.

    Returns: (str): restraint label.
    """
    
    return receptor + "_" + ligand


def get_XL_restraint_label(receptor, ligand):
    """
    Get label for inter-receptor-ligand crosslink restraint.
    
    Args:
    receptor (str): Name of receptor molecule.
    
    ligand (str): Name of ligand molecule.

    Returns:
    (str): restraint label.
    """
    
    return receptor + "_" + ligand


def get_escape_mutation_distance_restraint_label(receptor, ligand,
                                                 receptor_residue):
    """
    Get label for inter-receptor-ligand escape mutation
    distance restraint label.
    
    Args:
    receptor (str): Name of receptor molecule.
    
    ligand (str): Name of ligand molecule.

    receptor_residue (int): Escape mutant residue on the receptor.

    Returns:
    (str): restraint label.
    """
    
    return receptor + ":" + str(receptor_residue) + "_" + ligand


def get_epitope_restraint_label(receptor, ligand):
    """
    Get label for inter-receptor-ligand epitope restraint label.
    
    Args:
    receptor (str): Name of receptor molecule.
    
    ligand (str): Name of ligand molecule.

    Returns:
    (str): restraint label.
    """
    
    return receptor + "_" + ligand


def get_ligand_pair_binding_restraint_label(receptor, ligand1, ligand2):
    """
    Get label for inter-ligand pair binding restraint.
    
    Args:
    receptor (str): Name of ligand molecule 1.
    
    ligand (str): Name of ligand molecule 2.

    Returns:
    (str): restraint label.
    """
    
    return receptor + ":" + ligand1 + "_" + ligand2


def read_XL_data(filename, r_offset=0):
    """
    Parse crosslink data from CSV file. Crosslink data files can only have
    four columns--
    <receptor_name>, <receptor_residue>, <ligand_name>, <ligand_residue>
    
    All crosslinks used in this work are assumed to be DSS crosslinks
    with cutoffs of 28-30 A unless mentioned otherwise.
    
    Args:
    filename (str): CSV file containing crosslink data.
    
    r_offset (int, optional): Offset between PDB file and FASTA sequence
    for the receptor, i.e. residue_id from PDB + offset = residue id from FASTA.
    Defaults to 0.

    Returns:
    (list): Tuples of the form
    (receptor name, receptor residue, ligand name, ligand residue id)
    The receptor residue is offset corrected.
    """
    
    df = pd.read_csv(filename)
    xls = []
    for i in range(len(df)):
        r, r_res, l, l_res = tuple(df.iloc[i])
        xls.append((r, r_res+r_offset, l, l_res))
    return xls


def read_escape_mutant_data(filename, r_offset=0):
    """
    Parse escape mutation data from CSV file. Escape mutation data files can only have two columns:
    <receptor_residue>, <ligand_residue_ranges>.
    
    The receptor residue key is an alphanumeric string for the escape
    mutant reisdue with the first character as the FASTA letter for that
    residue, while the remaining characters make up the (int) residue id. 
    
    **If the mutant residue is missing, write NONE
    in all caps as the mutant residue key in the first field.**
    
    The ligand residue ranges is a string of the form
    "[(b1, e1), (b2, e2), ...]" where (b_i, e_i) is the beginning and end 
    residue id of the ith segment from the ligand (nanobody) which is proximal
    to the receptor mutant residue.
    
    Args:
    filename (str): CSV file containing escape mutation data.
    
    r_offset (int, optional): Offset between PDB file and FASTA sequence
    for the receptor, i.e. residue_id from PDB + offset = residue id from FASTA.
    Defaults to 0.

    Returns:
    (list): List of tuples of the the form 
    (receptor residue name, receptor escape mutant residue, ligand residues)
    The receptor residue is offset corrected. If escape mutants are missing
    for a ligand, a None is returned in place of the first tuple.
    """
    
    out = []
    df = pd.read_csv(filename)
    for i in range(len(df)):
        this_df = df.iloc[i]
        
        rr = this_df["receptor_residue"]
        receptor_resname = None
        receptor_resid = None
        
        if rr != "NONE":
            receptor_resname = rr[0].upper()
            receptor_resid = int(rr[1:]) + r_offset
        
        l_resranges = eval(this_df["ligand_residue_ranges"])
        ligand_resids = []
        for lr in l_resranges:
            this_resids = list(range(lr[0], lr[1]+1))
            ligand_resids.extend(this_resids)   

        out.append((receptor_resname, receptor_resid, ligand_resids))
    
    return out


def read_ligand_pair_binding_data(filename):
    """
    Parse pairwise ligand (nanobody) binding data from CSV file. Pairwise
    binding data may have only three columns:
    <ligand 1 name>, <ligand 2 name>, <bind together or not?
    where the last column is a boolean (0 or 1). Ligands that bind together (1)
    share epitopes and/or have spatial overlaps. Ligands that don't bind 
    together (0) have distinct overlaps and excluded volume interactions enforced between them.
    
    Args:
    filename (str): CSV file containing inter-ligand competitive binding data.

    Returns:
    (list): Tuples of the form (ligand 1 name, ligand 2 name, bind together?)
    """
    
    out = []
    df = pd.read_csv(filename)
    for i in range(len(df)):
        this_df = df.iloc[i]
        l1, l2 = this_df["ligand1"], this_df["ligand2"]
        bind = this_df["bind"]
        out.append((l1, l2, bind))
    return out


def make_histogram(samples, nbins=25, sample_range=None):
    """
    Wrapper on numpy's 1D histogram function.
    
    Args:
    samples (array-like): List or array of sample values. (floats)
    
    nbins (int, optional): Number of histogram bins. Defaults to 25.
    
    sample_range (tuple, optional): (min, max) values of samples. Defaults to
    None, in which case the min and max values are calculated from the provided
    list of samples.

    Returns:
    (tuple): (centers, counts) of histogram bins. 
    """
    
    if sample_range is None:
        sample_range = (np.min(samples)*0.98, np.max(samples)*1.02)
    bin_vals, bin_edges = np.histogram(samples, bins=nbins, range=sample_range)
    bin_centers = [0.5*(bin_edges[i]+bin_edges[i+1]) for i in range(nbins)]
    return bin_centers, bin_vals


class TopologyHandler:
    """
    Wrapper on PMI's topology parser. Adds custom attribute dictionaries
    for receptor and ligands.
    """
    
    def __init__(self, topology_file, pdb_dir=".", fasta_dir="."):
        """
        Constructor.
        
        Args:
        topology_file (str): PMI topology file.
        
        pdb_dir (str, optional): Directory containing all pdb files used in modeling and mentioned in the topology file. Defaults to ".".
        
        fasta_dir (str, optional): Directory containing all FASTA files used in modeling and mentioned in the topology file. Defaults to ".".
        """
        
        t = IMP.pmi.topology.TopologyReader(topology_file,
                                            pdb_dir=pdb_dir,
                                            fasta_dir=fasta_dir)
        self.PMI_topology = t
        self.components = t.get_components()
        self.pdb_dir = pdb_dir
        self.fasta_dir = fasta_dir
        
        self.molinfo = OrderedDict()
        
        # receptor
        self.receptor = None
        self.receptor_offset = 0
        self.num_receptor_copies = 0
        self._parse_receptor()
        
        # ligands
        self.ligands = []
        self._parse_ligands()
        
    def _parse_receptor(self):
        """
        Parse receptor information, and extract the number of receptor copies.
        Also set-up output RMF and PDB chain names for each receptor copy.
        """
        
        receptor = self.components[0].molname
        num_receptor_copies = 1
        for c in self.components[1:]:
            if c.molname == receptor and c.copyname:
                num_receptor_copies += 1
        receptor_offset = self.components[0].pdb_offset

        # check that the n_copies of the receptor are mentioned up front 
        # in the list of topology components
        # and that all of them have the same offset
        assert all([c.molname == receptor and c.pdb_offset == receptor_offset
                    for c in self.components[1:num_receptor_copies]])
        
        receptor_components = self.components[:num_receptor_copies]
        for i, c in enumerate(receptor_components):
            key = c.molname + "." + str(i)
            c.copyname = str(i)
            info = {"src_pdb_file": c.pdb_file,
                    "src_pdb_chain": c.chain,
                    "tar_pdb_chain": str(i),
                    "tar_rmf_chain": string.ascii_uppercase[i],
                    "offset": c.pdb_offset,
                    "color": "".join(c.color.split())}
            
            # very rudimentary way to account for the receptor residues
            # TODO: add checks to this, to allow for cases where
            # the range tuple is specified as (1,END) etc.
            # also remove representation for flexible residues in between
            # the extremities of the residue range given in the 
            # receptor topology
            resrange = c.residue_range
            info["src_pdb_residues"] = list(range(resrange[0], resrange[1]+1))
            
            self.molinfo[key] = info
        
        self.receptor = receptor
        self.receptor_offset = receptor_offset
        self.num_receptor_copies = num_receptor_copies        
        
    def _parse_ligands(self):
        """
        Parse information about each ligand, and set up output RMF and PDB 
        chain names for each ligand.
        """
        
        ligands = []
        for i, c in enumerate(self.components[self.num_receptor_copies:]):
            # assert that these all have zero offset
            # if not, the PDB & sequences should be made consistent
            assert c.pdb_offset == 0
            key = c.molname
            ligands.append(key)
            info = {
    "src_pdb_file": c.pdb_file,
    "src_pdb_chain": c.chain,
    "tar_pdb_chain": string.ascii_uppercase[i],
    "tar_rmf_chain": string.ascii_uppercase[self.num_receptor_copies+i],
    "color": "".join(c.color.split())}
            self.molinfo[key] = info
            self.ligands = ligands


class RMFHandler:
    """
    Wrapper that consolidates common RMF operations such as particle extraction
    and frame looping.
    """
    
    def __init__(self, rmf_file, read_freq=1):
        """
        Constructor.
        
        Args:
        rmf_file (str): RMF file name.
        
        read_freq (int, optional): Read every <read_freq>th frames only.
        Defaults to 1.
        """
        
        self.rmf_file = rmf_file
        self.file_handle = RMF.open_rmf_file_read_only(rmf_file)
        self._model = IMP.Model()
        
        hiers = IMP.rmf.create_hierarchies(self.file_handle, self._model)
        self.hierarchy = hiers[0]
    
        max_frames = self.file_handle.get_number_of_frames()
        self.frames = list(range(0, max_frames, read_freq))
    
    def get_particle(self, molecule, residue, copy_number=None,
                     has_radius=False):
        """
        Get a single particle corresponding to a single residue.
        
        Args:
        molecule (str): Molecule (chain) name to which the particle belongs.
        
        residue (int): Residue id of particle. May not be zero.
        
        copy_number (int, optional): Copy number of the molecule the particle
        belong to. Defaults to None, in which case a copy number of 0 is used.
        
        has_radius (bool, optional): Setting this as True builds spherical
        particles, i.e. decorated with IMP.core.XYZR instead of IMP.core.XYZ.
        Defaults to False.

        Returns:
        (IMP.Particle): IMP.core.XYZ or IMP.core.XYZR decorated particle.
        """
        
        self._model.update()
        if copy_number is not None and copy_number > 0:
            sel = IMP.atom.Selection(self.hierarchy, resolution=1,
                                     molecule=molecule, residue_index=residue,
                                     copy_index=copy_number)
        else:
            sel = IMP.atom.Selection(self.hierarchy, resolution=1,
                                     molecule=molecule, residue_index=residue)
        p = sel.get_selected_particles()[0]
        if has_radius:
            return IMP.core.XYZR(p)
        else:
            return IMP.core.XYZ(p)
    
    def get_particles(self, molecule, copy_number=None, has_radius=False):
        """
        Get all particles corresponding to a single molecule (chain).
        
        Args:
        molecule (str): Molecule (chain) name.
        
        copy_number (int, optional): Copy number of the molecule. Defaults to
        None, in which case a copy number of 0 is used.
        
        has_radius (bool, optional): Setting this as True builds spherical
        particles, i.e. decorated with IMP.core.XYZR instead of IMP.core.XYZ.
        Defaults to False.

        Returns:
        (list): All IMP.core.XYZ or IMP.core.XYZR decorated particles from the
        requested molecule (chain).
        """
        
        self._model.update()
        if copy_number is not None and copy_number >= 0:
            sel = IMP.atom.Selection(self.hierarchy, resolution=1,
                                     molecule=molecule, copy_index=copy_number)
        else:
            sel = IMP.atom.Selection(self.hierarchy, resolution=1, 
                                     molecule=molecule)
        if has_radius:
            ps = [IMP.core.XYZR(p) for p in sel.get_selected_particles()]
        else:
            ps = [IMP.core.XYZ(p) for p in sel.get_selected_particles()]
        return ps
    
    def load_frame(self, x):
        """
        Load a specified RMF frame.
        
        Args:
        x (int): Frame number.
        """
        
        self._model.update()
        IMP.rmf.load_frame(self.file_handle, x)
    
    def update(self):
        """
        Updates the underlying IMP.Model object to maintain consistency
        between different calls to the particles and/or during frame looping.
        """
        
        self._model.update()


def get_surface(ps, 
                probe_radius=5.0, surface_thickness=4.0,
                query_indices=[], query_radius=5.0, 
                render=False):
    """
    Get particles on the surface and core of a sphere cloud, whose constituent
    particles are given.

    Args:
    ps (list): IMP particles describing the sphere cloud.
    
    probe_radius (float, optional): Connolly surface construction
    algorithm probe radius. Defaults to 5.0.
    
    surface_thickness (float, optional): Thickness of the surface
    layer. Defaults to 4.0.
    
    query_indices (list): Indices of IMP particles of the sphere cloud,
    that are centers of the output surface. Defaults to empty.
    
    query_radius (float, optional): Radius from the query particles within
    which particles of the sphere cloud can be considered part of the 
    output surface. This makes sense only when a query indices
    have been provided. Defaults to 4.0
    
    render (bool, optional): If true, writes out BILD script in the
    current directory, for rendering with Chimerax. Defaults to False.
    
    Returns:
    (tuple): Lists of list indices of the particle list belonging to the surface
    and core, respectively, of the sphere cloud.
    """
    
    # get Connolly surface of the receptor
    spheres, centers = [], []
    for p in ps:
        p_xyzr = IMP.core.XYZR(p)
        c, r = p_xyzr.get_coordinates(), p_xyzr.get_radius()
        centers.append(c)
        spheres.append(IMP.algebra.Sphere3D(c,r))
        
    csplist = IMP.algebra.get_connolly_surface(spheres,
      _CONNOLLY_SURFACE_SAMPLING_DENSITY, probe_radius)
    connolly_points_ = [p.get_surface_point() for p in csplist]
    
    # filter the connolly points to ones that are within query radius
    # of query indices
    if query_indices:
        nn = IMP.algebra.NearestNeighbor3D(connolly_points_)
        connolly_indices = set()
        for i in query_indices:
            q = centers[i]
            this_ci = nn.get_in_ball(q, query_radius)
            connolly_indices |= set(this_ci)
        connolly_points = [connolly_points_[i] for i in connolly_indices]
    else:
        connolly_points = connolly_points_
    
    # get indices of surface particles
    surface_indices = []
    nn = IMP.algebra.NearestNeighbor3D(connolly_points)
    for i, c in enumerate(centers):
        if len(nn.get_in_ball(c, surface_thickness)):
            surface_indices.append(i)
    core_indices = [i for i in range(len(ps))
                    if i not in surface_indices]
        
    # write BILD script
    if render:
        s = ".color purple\n"
        for p in connolly_points:
            s += ".dot %2.2f %2.2f %2.2f\n" % tuple(p)
        s += "\n"
        
        # color surface spheres
        s += ".color salmon\n"
        for i in surface_indices:
            center = spheres[i].get_center()
            radius = spheres[i].get_radius() * 1.02
            s += ".sphere %2.2f %2.2f %2.2f %2.2f\n" % (*tuple(center),
                                                        radius)
        
        # color center indices differently, if they were supplied
        if query_indices:
            s += "\n"
            s += ".color red\n"
            for i in query_indices:
                center = spheres[i].get_center()
                radius = spheres[i].get_radius() * 1.04
                s += ".sphere %2.2f %2.2f %2.2f %2.2f\n" % (*tuple(center),
                                                        radius)
        s += "\n"
        with open("surface.bld", "w") as of:
            of.write(s)
    
    return surface_indices, core_indices


def set_initial_ligand_position_for_binary_docking(dof, dist=0.0):
    rb_receptor, rb_ligand = dof.get_rigid_bodies()
    
    # translate to center of receptor
    target_center = rb_receptor.get_coordinates() + \
                    dist * IMP.algebra.get_random_vector_on_unit_sphere()
    tr1 = target_center - rb_ligand.get_coordinates()
    IMP.core.transform(rb_ligand, tr1)
    
    # provide a random rotation
    rot = IMP.algebra.get_random_rotation_3d()
    tr2 = IMP.algebra.get_rotation_about_point(target_center, rot)
    IMP.core.transform(rb_ligand, tr2)
    
    