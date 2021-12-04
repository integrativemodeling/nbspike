import os
import itertools
import pandas as pd
import numpy as np
from collections import OrderedDict

from .base import BaseSatisfaction
from .. import epitopelib
from .. import utils
from ..graphics import plot_2d, plot_chimerax


# numerical constants
# epsilon for adjusting denominators to prevent underflow
EPS = 1e-15

# float format for saving stuff to disk.
SAVE_FLOAT_FMT = "%3.2f"


class Epitope(BaseSatisfaction):
    """ 
    Receptor epitope statistics calculator.
    """
    
    def setup(self, cutoff=8.0):
        """
        Set up this calculator. Get RMF particles corresponding to all receptor
        and ligand residues.

        Args:
        cutoff (float, optional): Max distance between a receptor and ligand
        residue to qualify as interfacial. Defaults to 8.0.
        """
        
        if cutoff is None: cutoff = DEFAULT_EPITOPE_CUTOFF
        self.cutoff = cutoff
        
        # extract particles
        ps, nps = [], []
        receptor_residues = []
        for i in range(self.num_receptor_copies):
            this_ps = self.rmf_handler.get_particles(self.receptor,
                                                     copy_number=i,
                                                     has_radius=True)
            ps.extend(this_ps)
            nps.append(len(this_ps))
            
            r = self.molinfo[self.receptor + "." + str(i)]["src_pdb_residues"]
            receptor_residues.append(r)
        
        for l in self.ligands:
            this_ps = self.rmf_handler.get_particles(l, has_radius=True)
            nps.append(len(this_ps))
            ps.extend(this_ps)
            
        self.particles = ps
        self.n_particles = np.array(nps, dtype=np.int32)
        self.receptor_residues = receptor_residues
        
        
    def run(self):
        """
        Calculate epitope probabilities for each receptor residue, 
        for each ligand. Also write UCSF ChimeraX scripts for rendering
        epitopes on receptor surface representations for each ligand.
        """
        
        print("\nCalculating receptor epitope overlap statistics...")
        # get coordinates
        coords = self._get_coords()
        
        # get epitopes
        epitopes = self._get_receptor_epitopes(coords)
        
        # make epitope histogram
        self._write_epitope_histogram(epitopes)
        
        # plot epitope histograms
        hist_fn = os.path.join(self.outdir, "epitope_histograms.csv")
        fig_fn = os.path.join(self.outdir, "epitope_histograms.png")
        ligand_colors = {l: self.molinfo[l]["color"] for l in self.ligands}
        
        plot_2d.plot_epitope_histogram(hist_fn, fig_fn, 
                                       self.receptor_residues,
                                       ligand_colors)
        
        # render chimerax scripts for rendering epitopes on the centroid
        try:
            centroid_idx = self.frames.index(self.centroid_frame)
        except IndexError:
            print("WARNING: Centroid frame not found, 3D epitops not rendered.")
            exit()
        centroid_epitopes = self._get_centroid_epitopes(epitopes, centroid_idx)
        plot_chimerax.render_epitope_maps(self.centroid_pdb_file, 
                        self.receptor, self.ligands, self.molinfo, 
                        self.num_receptor_copies,centroid_epitopes,
                        self.outdir)
    

    def render_binary_docking_epitope(self, XL_file=None,
                                    escape_mutant_file=None,
                                    target_receptor_copies=[0]):
        print("\nQuick visual consistency check of the epitope for binary docking by rendering it in ChimeraX")

        assert len(self.ligands) == 1
        ligand = self.ligands[0]

        outfile = os.path.join(self.outdir, 
                            ligand + "_binary_docking_epitope.cxc")

        plot_chimerax.render_binary_docking_epitope(self.centroid_pdb_file,
                   self.receptor, ligand, self.molinfo,
                   self.num_receptor_copies, outfile,
                   target_receptor_copies, XL_file, escape_mutant_file)    

   
    def _get_coords(self):
        """
        Get coordinates from RMF particles of parsed receptor and ligands.
        
        Returns:
        (3D numpy array of np.float32s): (X,Y,Z) coordinates of all particles
        (receptor copies first followed by all ligands) for all relevant
        frames from the given RMF file.
        """
        
        coords = []
        for i in range(len(self.frames)):
            self.rmf_handler.load_frame(self.frames[i])
            this_coords = [np.array(p.get_coordinates()) 
                           for p in self.particles]
            coords.append(this_coords)
        coords = np.array(coords, dtype=np.float32)
        return coords
    
    def _get_receptor_epitopes(self, coords):
        """
        Get epitope residues of the receptor for each docked ligand.
        
        Args:
        coords (3D numpy array of np.float32s): (X,Y,Z) coordinates of all
        particles (receptor copies first followed by all ligands) for all
        relevant frames from the given RMF file.

        Returns:
        (dict): Dict where each entry is a binary indicator 2D array across
        all RMF frames along rows, and the receptor sequence across columns. Each column is 1 or 0 depending on whether that receptor residue is a epitope for the ligand or not.
        """
        
        radii = np.array([p.get_radius() for p in self.particles], 
                         dtype=np.float32)
        
        epitope_pairs = epitopelib.get_epitope_pair(coords, radii, 
                                                    self.cutoff,
                                                    self.n_particles,
                                                    self.num_receptor_copies,
                                                    self.num_omp_threads)
        
        nr = sum(self.n_particles[:self.num_receptor_copies])
        nl = sum(self.n_particles) - nr
        nps_ligands = self.n_particles[self.num_receptor_copies:]
        epitope_pairs = epitope_pairs.reshape(-1, nr, nl)
        
        receptor_epitopes = OrderedDict()
        for i, l in enumerate(self.ligands):
            start = sum(nps_ligands[:i])
            stop = start + nps_ligands[i]
            e = np.max(epitope_pairs[:, :, start:stop], axis=2)
            receptor_epitopes[l] = e.astype(np.float32)
            
        return receptor_epitopes
    
    def _get_centroid_epitopes(self, epitopes, idx): 
        receptor_residue_tuple = []
        for i, rr in enumerate(self.receptor_residues):
            for r in rr:
                receptor_residue_tuple.append((i, r))
        
        centroid_epitopes = OrderedDict()
        for i, l in enumerate(self.ligands):
            indices = [np.bool(x) for x in epitopes[l][idx]]
            e = itertools.compress(receptor_residue_tuple, indices)
            centroid_epitopes[l] = list(e)
        return centroid_epitopes
    
    def _write_epitope_histogram(self, epitopes):
        """
        Write receptor probabilities to a CSV file.

        Args:
        epitopes (dict): Dict where each entry is a binary indicator 2D array
        across all RMF frames along rows, and the receptor sequence across
        columns. Each column is 1 or 0 depending on whether that receptor
        residue is a epitope for the ligand or not.
        """
        
        # compute histograms
        e = [[r for rr in self.receptor_residues for r in rr]]
        for l in self.ligands:
            this_e = np.mean(epitopes[l], axis=0)
            e.append(this_e)
        histograms = list(zip(*e))
        
        # write histograms to file
        colnames = ["receptor_residue"] + self.ligands
        df = pd.DataFrame(histograms, columns=colnames)            
        hist_fn = os.path.join(self.outdir, "epitope_histograms.csv")
        df.to_csv(hist_fn, index=False, float_format=SAVE_FLOAT_FMT)
        
