"""
Physics and data-driven spatial restraints for integrative modeling of 
nanobodies binding to the SARS-CoV2 Spike protein. Nanobodies are referred to
as ligands.
"""

import numpy as np
import itertools
from collections import OrderedDict

import IMP
import IMP.algebra
import IMP.atom
import IMP.container
import IMP.core
import IMP.pmi.tools
from IMP.pmi.restraints import RestraintBase

from . import utils


# numerical constants
# epsilon for adjusting denominators to prevent underflow
EPS = 1e-15

# slack for IMP close pair containers
# see: https://integrativemodeling.org/2.14.0/doc/ref/classIMP_1_1container_1_1CloseBipartitePairContainer.html#afa0bdd7250c318333de927ea873b11c6
SLACK = 10.0

# Connolly surface params
_COARSE_GRAINED_CONNOLLY_PROBE_RADIUS = 5.0 # A
_COARSE_GRAINED_CONNOLLY_SURFACE_THICKNESS = 4.0 # A


class CrosslinkDistanceRestraint(RestraintBase):
    """
    Crosslink distance restraint (implemented as an upper bound harmonic
    function) that attempts to keep the (sphere?) distance between 
    crosslinked particles lower than the linker spacer arm cutoff.
    
    This restraint actually maintains a set of distance restraints for each
    crosslink in a supplied crosslink dataset. The total score is reported
    as the sum of scores of each crosslink.
    """
    
    def __init__(self, root_hier, XL_pairs, resolution=1, receptor_copy=0,
                 kappa=1.0, cutoff=28.0, weight=1.0, label=None):
        """
        Constructor.
        
        Args:
        root_hier (IMP.hierarchy): Root hierarchy that provides access to all
        particles in the system.
        
        XL_pairs (list): List of crosslinked tuples of the form:
        (receptor name, receptor residue, ligand name, ligand residue)
        
        resolution (float, optional): Coarse grained resolution. Defaults to 1
        residue per bead.
        
        receptor_copy (int, optional): Copy number of receptor to target,
        for multimeric receptors. Defaults to 0.
        
        kappa (float, optional): Strength of the restraint.
        Defaults to 1.0 A^-2.
        
        cutoff (float, optional): Max stretched length of the crosslinker
        spacer arm. Defaults to 28.0 A. (typical values for a DSS crosslinker
        is 28-30 A)
        
        weight (float, optional): Relative weight of this restraint relative to
        other restraints in the system. Defaults to 1.0.
        
        label ([type], optional): Lable for this restraint. Defaults to None.
        """

        nxls = len(XL_pairs)
        
        receptor_name = XL_pairs[0][0]
        ligand_name = XL_pairs[0][2]
        assert all([XL_pairs[i][0] == receptor_name for i in range(nxls)])
        assert all([XL_pairs[i][2] == ligand_name for i in range(nxls)])
        
        print("\nGenerating %d crosslink distance restraints between molecules %s and %s" % (nxls, receptor_name, ligand_name))
        
        # get particle_pairs
        ppairs = []
        for xl in XL_pairs:
            receptor, receptor_residue, ligand, ligand_residue = xl
            sel_receptor = IMP.atom.Selection(root_hier, resolution=resolution,
                                            molecule=receptor,
                                            residue_index=receptor_residue,
                                            copy_index=receptor_copy)
            p_receptor = sel_receptor.get_selected_particles()[0]
            
            sel_ligand = IMP.atom.Selection(root_hier, resolution=resolution,
                                            molecule=ligand,
                                            residue_index=ligand_residue)
            p_ligand = sel_ligand.get_selected_particles()[0]
            
            ppairs.append((p_receptor, p_ligand))
        
        # init parent class
        model = ppairs[0][0].get_model()
        name = "CrosslinkDistance_%s" % label
        super().__init__(model, name=name, label=label, weight=weight)
          
        # add a single distance restraint for each crosslink pair
        ub = IMP.core.HarmonicUpperBound(cutoff, kappa)
        dps = IMP.core.DistancePairScore(ub)
        
        for i in range(nxls):
            #this_restraint = IMP.core.DistanceRestraint(model, ub, *ppairs[i])
            this_restraint = IMP.core.PairRestraint(model, dps, ppairs[i])
            self.rs.add_restraint(this_restraint)    
        
        self._include_in_rmf = True
    
    def get_output(self):
        """
        Overloaded get_output() method of IMP.pmi.restraint.RestraintBase
        that decides what gets output to stat files.
        
        Returns:
        (dict): Dictionary of outputs for stat files.
        """
        
        output = {}
        score = self.evaluate()
        output["_TotalScore"] = str(score)
        output["CrosslinkDistanceScore_" + self.label] = str(score)
        return output


class AmbiguousCrosslinkDistanceRestraint(RestraintBase):
    """
    Ambiguous Crosslink distance restraint (implemented as an upper bound
    harmonic function) that attempts to keep the (sphere?) distance between 
    crosslinked particles lower than the linker spacer arm cutoff.
    
    This restraint actually maintains a set of distance restraints for each
    crosslink in a supplied crosslink dataset. The total score is reported
    as the sum of scores of each crosslink.
    
    This restraint takes into account chain ambiguity in a greedy fashion.
    If a crosslinked residue is present in multiple copies of its parent 
    molecule, the copy that is actually used to calculate and record the score,
    is the one that produces the lowest score among other copies.
    """
        
    def __init__(self, root_hier, XL_pairs, resolution=1, receptor_copies=[], 
                 kappa=1.0, cutoff=28.0, weight=1.0, label=None):
        """
        Constructor.
        
        Args:
        root_hier (IMP.hierarchy): Root hierarchy that provides access to all
        particles in the system.
        
        XL_pairs (list): List of crosslinked tuples of the form:
        (receptor name, receptor residue, ligand name, ligand residue)
        
        resolution (float, optional): Coarse grained resolution. Defaults to 1
        residue per bead.
        
        receptor_copies (list, optional): Copy numbers of multimeric receptors.
        Default is empty list, in which all case all copies are selected.
        
        kappa (float, optional): Strength of the restraint.
        Defaults to 1.0 A^-2.
        
        cutoff (float, optional): Max stretched length of the crosslinker
        spacer arm. Defaults to 28.0 A. (typical values for a DSS crosslinker
        is 28-30 A)
        
        weight (float, optional): Relative weight of this restraint relative to
        other restraints in the system. Defaults to 1.0.
        
        label ([type], optional): Lable for this restraint. Defaults to None.
        """
        
        nxls = len(XL_pairs)
        
        receptor_name = XL_pairs[0][0]
        ligand_name = XL_pairs[0][2]
        assert all([XL_pairs[i][0] == receptor_name for i in range(nxls)])
        assert all([XL_pairs[i][2] == ligand_name for i in range(nxls)])
        
        print("\nGenerating %d crosslink distance restraints between molecules %s and %s" % (nxls, receptor_name, ligand_name))
        
        # init parent class
        model = root_hier.get_model()
        name = "CrosslinkDistance_%s" % label
        super().__init__(model, name=name, label=label, weight=weight)
        
        # create class attributes so that they can be called from
        # helper methods later
        self.model = model
        self.root_hier = root_hier
        self.resolution = resolution
        self.kappa = kappa
        self.cutoff = cutoff
        self.receptor_copies = receptor_copies

        # add a min. distance restraint for each crosslink
        self.pi_pairs = OrderedDict()
        self.model = model
        for xl in XL_pairs:
            self._create_crosslink_restraint(xl)
        self._include_in_rmf = True
        
    
    def _create_crosslink_restraint(self, xl):
        """
        Create a single crosslink distance restraint for a given crosslink.
        
        Args:
        xl (tuple): Tuple of the form:
        (receptor name, receptor residue, ligand name, ligand residue)
        """
        
        # parse the crosslinked residue pair
        receptor, receptor_residue, ligand, ligand_residue = xl
        
        # get receptor particles
        receptor_args = {"hierarchy": self.root_hier,
                        "resolution": self.resolution,
                        "molecule": receptor,
                        "residue_index": receptor_residue}
        
        if self.receptor_copies:
            receptor_args["copy_indexes"] = self.receptor_copies

        sel_receptor = IMP.atom.Selection(**receptor_args)
        ps_receptor = sel_receptor.get_selected_particles()
        
        # get ligand particles
        sel_ligand = IMP.atom.Selection(self.root_hier,
                                        resolution=self.resolution,
                                        molecule=ligand,
                                        residue_index=ligand_residue)
        ps_ligand = sel_ligand.get_selected_particles()
        
        # initialize a distnace pair score
        ub = IMP.core.HarmonicUpperBound(self.cutoff, self.kappa)
        dps = IMP.core.DistancePairScore(ub)
        #dps = IMP.core.HarmonicUpperBoundSphereDistancePairScore(self.cutoff,
        #                                                         self.kappa)
        
        # create a table refiner for receptor and ligand selections
        tref = IMP.core.TableRefiner()
        tref.add_particle(ps_receptor[0], ps_receptor)
        tref.add_particle(ps_ligand[0], ps_ligand)
        
        # create closest pair score that wraps the distance pair score
        sf = IMP.core.KClosePairsPairScore(dps, tref, 1)
        
        # create an underlying pair restraint see the implementation in:
        # https://integrativemodeling.org/2.14.0/doc/ref/core_2restrain_minimum_distance_8py-example.html
        pi_pair = (ps_receptor[0], ps_ligand[0])
        restraint = IMP.core.PairRestraint(self.model, sf, pi_pair)
        self.rs.add_restraint(restraint)
    
    def get_output(self):
        """
        Overloaded get_output() method of IMP.pmi.restraint.RestraintBase
        that decides what gets output to stat files.
        
        Returns:
        (dict): Dictionary of outputs for stat files.
        """
        
        output = {}
        score = self.evaluate()
        output["_TotalScore"] = str(score)
        output["CrosslinkDistanceScore_" + self.label] = str(score)
        return output
    
    
class EpitopeRestraint(RestraintBase):
    def __init__(self, root_hier,
                receptor_name, ligand_name, ligand_residues=[],
                receptor_copies=[], resolution=1,
                epitope_center_residues=[], epitope_cutoff=4.0,
                kappa=1.0, cutoff=4.0, 
                connolly_surface_params=None,
                weight=1.0, label=None):

        print("\nGenerating a new epitope restraint between receptor %s and ligand %s" % (receptor_name, ligand_name))
        
        # get particles corresponding to the entire receptor
        kwargs = {"hierarchy": root_hier,
                  "resolution": resolution,
                  "molecule": receptor_name}
        if receptor_copies:
            if not isinstance(receptor_copies, list):
                receptor_copies = [receptor_copies]
            kwargs["copy_indexes"] = receptor_copies
        sel = IMP.atom.Selection(**kwargs)
        ps = sel.get_selected_particles()
        
        # get particles corresponding to the query receptor residues
        query_indices = []
        if epitope_center_residues:
            kwargs["residue_indexes"] = epitope_center_residues
            sel = IMP.atom.Selection(**kwargs)
            query_ps = sel.get_selected_particles()
            query_indices = [ps.index(p) for p in query_ps]
            
        # get particles on the receptor surface in the vicinity
        # of the given receptor particles
        if connolly_surface_params is not None:
            r, t = connolly_surface_params
        else:
            r = _COARSE_GRAINED_CONNOLLY_PROBE_RADIUS
            t = _COARSE_GRAINED_CONNOLLY_SURFACE_THICKNESS
        surface_indices, _ = utils.get_surface(ps,
                                               query_indices=query_indices, 
                                               probe_radius=r,
                                               surface_thickness=t,
                                               query_radius=epitope_cutoff,
                                               render=True)
        ps_receptor = [ps[i] for i in surface_indices]
       
        # get particles corresponding to ligand residues
        if ligand_residues:
            sel = IMP.atom.Selection(root_hier, resolution=resolution,
                                 molecule=ligand_name,
                                 residue_indexes=ligand_residues)
        else:
            sel = IMP.atom.Selection(root_hier, resolution=resolution,
                                     molecule=ligand_name)
        ps_ligand = sel.get_selected_particles()
        
        # init parent class
        name = "EpitopeScore_%s" % label
        self.model = ps_receptor[0].get_model()
        super().__init__(self.model, name=name, label=label, weight=weight)
        
        # close bipartite container between receptor and ligand
        lsr = IMP.container.ListSingletonContainer(self.model)
        lsr.add(ps_receptor)
        
        lsl = IMP.container.ListSingletonContainer(self.model)
        lsl.add(ps_ligand)
        
        cpc = IMP.container.CloseBipartitePairContainer(
            lsr, lsl, 20.0, SLACK)
        
        # pair score
        dps = IMP.core.HarmonicUpperBoundSphereDistancePairScore(cutoff, kappa)
        
        # restraint
        restraint = IMP.container.PairsRestraint(dps, cpc)
        self.rs.add_restraint(restraint)
        
        self._include_in_rmf = True
    
    def get_output(self):
        output = {}
        score = self.evaluate()
        output["TotalScore"] = str(score)
        output["EpitopeScore_" + self.label] = str(score)
        return output
            
                        
class EscapeMutationDistanceRestraint(RestraintBase):
    """
    Distance Restraint for the minimum distance between an escape mutant 
    residue on the receptor, and a set of residues (usually the CDR3 loop)
    on the ligand.
    """
    
    def __init__(self, root_hier, 
                 receptor_name, ligand_name,
                 receptor_residue, ligand_residues=[],
                 receptor_copies=[], resolution=1,
                 kappa=1.0, cutoff=8.0,
                 weight=1.0, label=None):
        """
        Constructor.
        
        Args:
        root_hier (IMP.hierarchy): Root hierarchy that provides access to all
        particles in the system.
        
        receptor_name (str): Name of receptor molecule (chain).
        
        ligand_name (str): Name of ligand molecule (chain).
        
        receptor_residue (int): Target residue on the receptor.
        
        ligand residues (list, optional): List of residues on the ligand.
        Defaults to empty list.
        
        receptor_copies (list, optional): Copy numbers of multimeric receptors.
        Default is empty list, in which all case all copies are selected.
        
        resolution (float, optional): Coarse grained resolution. Defaults to 1
        residue per bead.
        
        kappa (float, optional): Strength of this restraint. Defaults to 1.0.
        
        cutoff (float, optional): Max. allowed value of the closest
        approach distance between receptor and ligand regions. 
        Defaults to 8.0 A.
        
        weight (float, optional): Relative weight of this restraint relative to
        other restraints in the system. Defaults to 1.0.
        
        label (str, optional): Label for this restraint. Defaults to None.
        """
        
        print("\nGenerating a new escape mutation distance restraint between residue %d of receptor %s and ligand %s" % (receptor_residue, receptor_name, ligand_name))
        
        # get particles corresponding to receptor residues
        kwargs = {"hierarchy": root_hier,
                  "resolution": resolution,
                  "molecule": receptor_name,
                  "residue_index": receptor_residue}
        if receptor_copies:
            if not isinstance(receptor_copies, list):
                receptor_copies = [receptor_copies]
            kwargs["copy_indexes"] = receptor_copies
        sel = IMP.atom.Selection(**kwargs)
        ps_receptor = sel.get_selected_particles()
        
        # get particles corresponding to ligand residues
        if ligand_residues:
            sel = IMP.atom.Selection(root_hier, resolution=resolution,
                                molecule=ligand_name,
                                residue_indexes=ligand_residues)
        else:
            sel = IMP.atom.Selection(root_hier, resolution=resolution,
                                     molecule=ligand_name)
        ps_ligand = sel.get_selected_particles()
        
        # init parent class
        name = "EscapeMutationDistanceScore_%s" % label
        self.model = ps_receptor[0].get_model()
        super().__init__(self.model, name=name, label=label, weight=weight)
        
        # initialize a distance pair score with a harmonic upper bound
        ub = IMP.core.HarmonicUpperBound(cutoff, kappa)
        dps = IMP.core.DistancePairScore(ub)
        
        # create a table refiner
        tref = IMP.core.TableRefiner()
        tref.add_particle(ps_receptor[0], [ps_receptor[0]])
        tref.add_particle(ps_ligand[0], ps_ligand)
        
        # create closest pair score that wraps the distance pair score
        sf = IMP.core.KClosePairsPairScore(dps, tref, 3)
        
        # create an underlying pair restraint see the implementation in:
        # https://integrativemodeling.org/2.14.0/doc/ref/core_2restrain_minimum_distance_8py-example.html
        pi_pair = (ps_receptor[0], ps_ligand[0])
        restraint = IMP.core.PairRestraint(self.model, sf, pi_pair)
        self.rs.add_restraint(restraint)
        
        self._include_in_rmf = True
    
    def get_output(self):
        """
        Overloaded get_output() method of IMP.pmi.restraint.RestraintBase
        that decides what gets output to stat files.
        
        Returns:
        (dict): Dictionary of outputs for stat files.
        """
        
        output = {}
        score = self.evaluate()
        output["_TotalScore"] = str(score)
        output["EscapeMutationDistanceScore_" + self.label] = str(score)
        return output
        
