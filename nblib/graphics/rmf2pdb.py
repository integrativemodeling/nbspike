"""
RMF to PDB file translator. 
"""

import os
import copy
import string
import numpy as np
from collections import OrderedDict

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB import Residue, Chain, Model, Structure
from Bio.SVDSuperimposer import SVDSuperimposer

import IMP.pmi.topology

from .. import utils

sup = SVDSuperimposer()

class RigidPDBWriter:
    """
    Translates from a RMF to a PDB file. Collects all the rigid bodies PDB
    structures used in modeling and aligns them with corresponding particles
    from a RMF file.
    """
    
    def __init__(self, rmf_file, topology_file, 
                 pdb_dir=".", fasta_dir=".",
                 frames=[], read_freq=1):
        """
        Constructor.

        Args:
        rmf_file (str): RMF filename.
        
        topology_file (str): PMI topology file.
        
        pdb_dir (str, optional): Directory containing all pdb files used in modeling and mentioned in the topology file. Defaults to ".".
        
        fasta_dir (str, optional): Directory containing all FASTA files used in modeling and mentioned in the topology file. Defaults to ".".
        
        frames (list, optional): List of RMF frames to convert. 
        Defaults to empty list, in which case all frames from the RMF file
        are read at the given read frequency.
        
        read_freq (int, optional): Frequency of RMF frame read operations,
        if no list of frames are supplied. Defaults to 1.
        """
        
        self.rmf_handler = utils.RMFHandler(rmf_file, read_freq=read_freq)
        if not frames:
            frames = self.rmf_handler.frames
        self.frames = frames
        
        # parse the topology file
        th = utils.TopologyHandler(topology_file, pdb_dir=pdb_dir,
                                   fasta_dir=fasta_dir)
        
        self.receptor = th.receptor
        self.ligands = th.ligands
        self.molinfo = th.molinfo
        self.num_receptor_copies = th.num_receptor_copies
        
        # add biopython models objects
        self.pdb_models = OrderedDict()
        for k, v in self.molinfo.items():
            pdb_file = os.path.join(pdb_dir, v["src_pdb_file"])
            m = PDBParser(QUIET=True).get_structure("x", pdb_file)[0]
            self.pdb_models[k] = m

        # get rmf particles
        ps = self._get_RMF_particles()
        self.rmf_ps_receptor = ps[0]
        self.rmf_ps_ligands = ps[1]
        
        # get pdb residues
        residues = self._get_PDB_residues()
        self.pdb_res_receptor = residues[0]
        self.pdb_res_ligands = residues[1]
        
    def set_frames(self, frames=None):
        """
        Specify a list of RMF frames. This ensures re-purposing one instance
        of this object for translating multiple frames, multiple times.

        Args:
        frames (list, optional): List of frames. Defaults to None.
        """
        
        if frames is None:
            return
        if not isinstance(frames, list):
            frames = [frames]
        self.frames = frames
    
    def write(self, out_pdb_file):
        """
        Main workhorse method that does the RMF to PDB conversion and writes
        the output file to disk.
        
        Args:
        out_pdb_file (str): Output PDB file name.
        """
        
        struct = Structure.Structure("x")
        
        for ii, i in enumerate(self.frames):
            # load new frame
            self.rmf_handler.load_frame(i)
            
            # align receptor rmf and pdb
            for i in range(self.num_receptor_copies):
                key = self.receptor + "." + str(i)
                self._align(self.rmf_ps_receptor[key], 
                            self.pdb_res_receptor[key])
            
            # align ligand rmf and pdb
            for l in self.ligands:
                self._align(self.rmf_ps_ligands[l],
                            self.pdb_res_ligands[l])
                
            # init a new model
            model = Model.Model(ii)
            
            # add receptor atoms
            for i in range(self.num_receptor_copies):
                key = self.receptor + "." + str(i)
                chain = Chain.Chain(self.molinfo[key]["tar_pdb_chain"])
                residues = copy.copy(self.pdb_res_receptor[key])
                [chain.add(r) for r in residues]
                model.add(chain)
            
            # add ligand atoms
            for l in self.ligands:
                chain = Chain.Chain(self.molinfo[l]["tar_pdb_chain"])
                residues = copy.deepcopy(self.pdb_res_ligands[l])
                [chain.add(r) for r in residues]
                model.add(chain)

            struct.add(model)
        
        io = PDBIO()
        io.set_structure(struct)
        io.save(out_pdb_file)

        
    def _get_RMF_particles(self):
        """
        Get all RMF particles for this system.
        
        Returns:
        (tuple): Dicts for receptor and ligand particles.
        """
        
        self.rmf_handler.update()
        
        # receptor
        ps_receptor = OrderedDict()
        for i in range(self.num_receptor_copies):
            ps = self.rmf_handler.get_particles(molecule=self.receptor,
                                                copy_number=i)
            ps_receptor[self.receptor + "." + str(i)] = ps
                
        # ligands
        ps_ligands = OrderedDict()
        for l in self.ligands:
            ps_ligands[l] = self.rmf_handler.get_particles(molecule=l)
        
        return ps_receptor, ps_ligands
    
    def _get_PDB_residues(self):
        """
        Get Biopython Residue objects corresponding to each residue from the
        original PDB files used as rigid bodies in integrative modeling.

        Returns:
        (tuple): Dicts for receptor and ligand Biopython residue objects.
        """
        
        # receptor
        res_receptor = OrderedDict()
        for i in range(self.num_receptor_copies):
            key = self.receptor + "." + str(i)
            model = self.pdb_models[key]
            chain = self.molinfo[key]["src_pdb_chain"]
            allowed_residues = self.molinfo[key]['src_pdb_residues']
            this_res_receptor = []
            for r in model[chain].get_residues():
                if (r.id[0] != " ") or (r.id[1] not in allowed_residues):
                    continue
                new_r = Residue.Residue(id=r.id, resname=r.resname, 
                                        segid=r.segid)
                [new_r.add(a) for a in r.get_atoms()]
                this_res_receptor.append(new_r)
            res_receptor[key] = this_res_receptor
            
        # ligands
        res_ligands = OrderedDict()
        for l in self.ligands:
            key = l
            model = self.pdb_models[key]
            chain = self.molinfo[key]["src_pdb_chain"]
            this_res_ligand = []
            for r in model[chain].get_residues():
                if r.id[0] != " ":
                    continue
                new_r = Residue.Residue(id=r.id, resname=r.resname, 
                                        segid=r.segid)
                [new_r.add(a) for a in r.get_atoms()]
                this_res_ligand.append(new_r)
            res_ligands[key] = this_res_ligand
        
        return res_receptor, res_ligands
    
    def _align(self, rmf_ps, pdb_res):
        """
        For a set of residues, align their RMF particles with their 
        corresponding Biopython residue objects. The CA atoms of the
        Biopython objects are aligned with the coarse-grained 1 residue
        per bead RMF particles. The alignment is then applied to each
        child atom of the Biopython residue objects.

        Args:
        rmf_ps (list): List of RMF particles.
        
        pdb_res (list): List of Biopython residue objects.
        """
        
        assert len(rmf_ps) == len(pdb_res)
        rmf_coords = np.array([p.get_coordinates() for p in rmf_ps])
        pdb_coords = np.array([r["CA"].coord for r in pdb_res])
        sup.set(rmf_coords, pdb_coords)
        sup.run()
        rotmat, vec = sup.get_rotran()
        pdb_atoms = [a for r in pdb_res for a in r.get_atoms()]
        [a.transform(rotmat, vec) for a in pdb_atoms]
        