import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import IMP.core

from .base import BaseSatisfaction
from .. import utils
from ..graphics import plot_2d


# float format for saving stuff to disk.
SAVE_FLOAT_FMT = "%3.2f"


class EscapeMutationSatisfaction(BaseSatisfaction):
    """
    Distance and satisfaction calculator for escape mutation distance
    restraints.
    """
    
    def add_data(self, data_files, cutoff=8.0):
        """
        Add all escape mutation datasets.
        
        This method parses IMP.core.XYZ decorated
        particles corresponding to escape mutant receptor residues and their
        corresponding complementary regions on the ligands.

        Args:
        data_files (dict): Dictof escape mutation data CSV files.
        Escape mutation data files can only have four columns:
        <mutant_residue_key>, <ligand_residue_ranges>.
    
        The mutant residue key is an alphanumeric string with the first
        character as the FASTA letter for that residue, while the remaining
        characters make up the (int) residue id.
        
        The ligand residue ranges is a string of the form
        "[(b1, e1), (b2, e2), ...]" where (b_i, e_i) is the beginning and end 
        residue id of the ith segment from the ligand (nanobody) which is
        proximal to the receptor mutant residue.
        
        cutoff (float, optional): Threshold center-to-center distance between
        receptor mutant residue and closest approach point on the ligand
        CDR3 region.
        """
        
        self.cutoff = cutoff

        # get particles for each dataset
        pdict = OrderedDict()
        ligands = set()
        for l, fn in data_files.items():
            parsed_data = utils.read_escape_mutant_data(fn,self.receptor_offset)
            
            for rn, rr, lrs in parsed_data:
                if (rn is None or rr is None): continue
                ps_r = []
                for  i in range(self.num_receptor_copies):
                    p = self.rmf_handler.get_particle(self.receptor, rr,
                                                copy_number=i,
                                                has_radius=False)
                    ps_r.append(p)

                ps_l = [self.rmf_handler.get_particle(l, i, has_radius=False)
                        for i in lrs]
                key = (rn + str(rr-self.receptor_offset), l)
                val = (ps_r, ps_l)
                
                ligands |= {l}
                pdict[key] = val
        
        self.particle_lookup = pdict
        
        # trim the set of ligands
        self.ligands = sorted(list(ligands))
       
       
    def run(self):
        """
        Calculate mean values of closest approach distance between receptor
        escape mutant residue and ligand.
        
        Restraint satisfaction is calculated as:
        i) avg. satisfaction: the fraction of times a crosslink is satisfied
        
        ii) ensemble satisfaction: a crosslink is considered satisfied
        if it is satisfied at least once in the ensemble of models. Ensemble
        here mostly refers to the members of a cluster.
        
        iii) centroid satisfaction: how many crosslinks are satisfied in the
        centroid / representative model from a cluster of models.
        """
        
        print("\nCalculating escape mutant residue distance statistics...")
        d_mean, d_std, avg_sat, ensemble_sat = self._get_statistics()
        d_centroid, centroid_sat = self._get_centroid_statistics() 
        
        distance_data = (d_mean, d_std, d_centroid)
        sat_data = (avg_sat, ensemble_sat, centroid_sat)
        self._write(distance_data, sat_data)
        
        data_fn = os.path.join(self.outdir, "summary_escape_mutation.csv")
        fig_fn = os.path.join(self.outdir, "escape_mutant_distances.png")
        ligand_colors = {l: self.molinfo[l]["color"] for l in self.ligands}
        plot_2d.plot_escape_mutation_distances(data_fn, fig_fn, 
                                               self.receptor, self.ligands,
                                               ligand_colors, self.cutoff)
    
    
    def _get_distances_frame(self):
        """
        Get closest distances from escape mutant receptor residues to 
        corresponding ligand.

        Args:
        ligand (str): Name of ligand molecule.

        Returns:
        (tuple): (distances, copy numbers).
        Copy numbers are of the receptor monomers for which the closest approach
        distance is a minimum across all receptor monomers / copies. This is
        used for ambiguous escape mutation distance restraints. 
        """
        
        distances = OrderedDict()
        receptor_copies = OrderedDict()
        for k, (ps_r, ps_l) in self.particle_lookup.items():
            d = []
            for p_r in ps_r:
                this_d = min([IMP.core.get_distance(p_r, p) for p in ps_l])
                d.append(this_d)
            distances[k] = min(d)
            receptor_copies[k] = d.index(min(d))
        return distances, receptor_copies
    
    def _get_centroid_statistics(self):
        """
        Get closest approach distances and corresponding satisfaction
        (i.e. whether those distances are less than imposed cutoff) for
        receptor escape mutants, for the centroid frame only.
        
        Returns:
        (tuple): (distances, satisfaction)
        """
        
        self.rmf_handler.load_frame(self.centroid_frame)
        distances, _ = self._get_distances_frame()
        
        sat = OrderedDict()
        for k, v in distances.items():
            sat[k] = float(v <= self.cutoff)
        return distances, sat
            
    def _get_statistics(self):
        """
        Calculate mean and std of receptor-ligand closest approach distances
        and satisfaction (avg and ensemble). 
        
        Returns:
        (tuple): (distances, avg sat, ensemble sat) for each distance restraint.
        """
        # result accumulators
        distances = OrderedDict()
        distances_mean = OrderedDict()
        distances_std = OrderedDict()
        avg_sat = OrderedDict()
        ensemble_sat = OrderedDict()
        
        for k in self.particle_lookup:
            distances[k] = []
            distances_mean[k] = 0.0
            distances_std[k] = 0.0
            
            avg_sat[k] = 0.0

        # frame iteration
        nframes = len(self.frames)
        for i in range(nframes):
            self.rmf_handler.load_frame(self.frames[i])
            this_distances, _ = self._get_distances_frame()
            for k, v in this_distances.items():   
                distances[k].append(v)
                avg_sat[k] += float(v <= self.cutoff)
        
        # reduce over all frames
        for k in self.particle_lookup:
            distances_mean[k] = np.mean(distances[k])
            distances_std[k] = np.std(distances[k], ddof=1)
            avg_sat[k] /= float(nframes)
            ensemble_sat[k] = float(avg_sat[k] > 0.0)
        
        return distances_mean, distances_std, avg_sat, ensemble_sat
        
    def _write(self, distance_data, sat_data):
        """
        Write computed statistics to file.
        Args:
        distance_data (tuple): (mean, std, centroid) values of closest
        approach distance for each escape mutation distance restraint.
        
        sat_data (tuple): (avg sat, ensemble sat, centroid sat) for
        each distance restraint.
        """
        
        d_mean, d_std, d_centroid = distance_data
        avg_sat, ensemble_sat, centroid_sat = sat_data
        
        df = []
        for k in self.particle_lookup:
            this_df = (*k, d_mean[k], d_std[k], d_centroid[k], 
                       avg_sat[k], ensemble_sat[k], centroid_sat[k])
            df.append(this_df)
        
        colnames = ["escape_mutant_residue", "ligand", 
                    "distance_mean", "distance_std", "distance_centroid",
                    "average_sat", "ensemble_sat", "centroid_sat"]
        df = pd.DataFrame(df, columns=colnames)
        outfn = os.path.join(self.outdir, "summary_escape_mutation.csv")
        df.to_csv(outfn, index=False, float_format=SAVE_FLOAT_FMT)
    
