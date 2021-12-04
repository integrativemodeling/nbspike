"""
Base class definitions for restraint satisfaction calculators.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from .. import utils

class BaseSatisfaction:
    """
    Class definition for a base class for restraint satisfcation calculators.
    Except the constructor, all other methods in this class will be heavily
    overloaded in subclasses corresponding to different restraints like
    crosslinks, escape mutations, etc. 
    """
    
    def __init__(self, topology_file, rmf_file, centroid_pdb_file,
                 pdb_dir=".", fasta_dir=".",
                 centroid_frame=0, frames=[],
                 outdir=".", readfreq=1, num_omp_threads=1):
        """
        Constructor
        
        Args:
        topology_file (str): PMI topology file.
        
        rmf_file (str): RMF filename containing all docked models.
        
        centroid_pdb_file (str): PDB file corresponding to the centroid
        of a particular cluster.
        
        pdb_dir (str, optional): Directory containing all pdb files used in modeling and mentioned in the topology file. Defaults to ".".
        
        fasta_dir (str, optional): Directory containing all FASTA files used in modeling and mentioned in the topology file. Defaults to ".".
        
        centroid_frame (int, optional): RMF frame corresponding to a cluster
        centroid, when processing the supplied list of frames are 
        members of a cluster. Defaults to 0.
        
        frames (list, optional): List of RMF frames to work with.
        Defaults to empty, in which case all frames from the RMF file are
        read at a given reading frequency.
        
        outdir (str, optional): Top level output directory for all results.
        Defaults to ".".
        
        readfreq (int, optional): Read frequency of frames from given
        RMF file. Defaults to 1.
        
        num_omp_threads (int, optional): Number of threads for parallel
        computation. Defaults to 1.
        """
        
        # parse topology file
        th = utils.TopologyHandler(topology_file, pdb_dir=pdb_dir,
                                   fasta_dir=fasta_dir)
        
        self.receptor = th.receptor
        self.ligands = th.ligands
        self.molinfo = th.molinfo
        self.num_receptor_copies = th.num_receptor_copies
        self.receptor_offset = th.receptor_offset
        
        # parse traj RMF file
        self.rmf_handler = utils.RMFHandler(rmf_file, read_freq=readfreq)
        
        # frames for the RMF file and cluster centroids (if given)
        if not frames:
            frames = self.rmf_handler.frames
        self.frames = frames
        self.centroid_frame = centroid_frame
        
        # parse centroid pdb file
        self.centroid_pdb_file = centroid_pdb_file
        
        # create output dir
        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir, exist_ok=True)
        
        # number of threads for parallel processing
        self.num_omp_threads = num_omp_threads
    
    
    def add_data(self, data_file, **kwargs):
        # defined in subclasses
        pass
        
    def run(self):
        # defined in subclasses
        pass
    