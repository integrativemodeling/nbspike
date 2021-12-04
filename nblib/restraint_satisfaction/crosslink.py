import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import IMP.core

from .base import BaseSatisfaction
from .. import utils
from ..graphics import plot_2d, plot_chimerax


# number of bins for distance histograms
NBINS = 25

# float format for saving stuff to disk.
SAVE_FLOAT_FMT = "%3.2f"


class CrosslinkSatisfaction(BaseSatisfaction):
    """
    Crosslink satisfaction and statistics calculator.
    """
    
    def add_data(self, data_files, cutoff=28.0):
        """
        Add all crosslink datasets. This method parses IMP.core.XYZR decorated
        particles corresponding to crosslinked residues.

        Args:
        data_files (dict): Dict of crosslink data files. Crosslink data files 
        are CSV format files and can only have four columns--
        <receptor_name>, <receptor_residue>, <ligand_name>, <ligand_residue>
        
        cutoff (float, optional): Max. stretched length of the spacer
        arm of the crosslinking reagent. Defaults to 28.0 A (typical for DSS).
        """
        
        if cutoff is None: cutoff = DEFAULT_DSS_CUTOFF
        self.cutoff = cutoff
        
        # get particles for each xl dataset
        pdict = OrderedDict()
        ligands = set()
        for l, fn in data_files.items():
            # get xl pairs
            xl_pairs = utils.read_XL_data(fn, r_offset=self.receptor_offset)
            
            # get particles corresponding to these xl pairs
            this_pdict = OrderedDict()
            for xl in xl_pairs:
                r_, r_res, l_, l_res = xl
                ps_r = []
                for  i in range(self.num_receptor_copies):
                    p = self.rmf_handler.get_particle(r_, r_res, copy_number=i,
                                                      has_radius=False)
                    ps_r.append(p)
                
                p_l = self.rmf_handler.get_particle(l_, l_res, has_radius=False)
                key = (r_, r_res-self.receptor_offset, l_, l_res)
                val = (ps_r, p_l)
                this_pdict[key] = val
                
            # run some quick checks
            assert all([k[0] == self.receptor for k in this_pdict]) 
            assert all([k[2] == l for k in this_pdict])
            ligands |= {l}
            
            # add to master dict    
            pdict[l] = this_pdict
        
        self.particle_pair_lookup = pdict
        
        # trim the set of ligands
        self.ligands = sorted(list(ligands))
        
    
    def run(self):
        """
        Calculate distance distributions and satisfcation metrics for each
        crosslink dataset. Satisfaction is calculated as:
        i) avg. satisfaction: the fraction of times a crosslink is satisfied
        
        ii) ensemble satisfaction: a crosslink is considered satisfied
        if it is satisfied at least once in the ensemble of models. Ensemble
        here mostly refers to the members of a cluster.
        
        iii) centroid satisfaction: how many crosslinks are satisfied in the
        centroid / representative model from a cluster of models.
        """
        
        # calculate crosslink statistics
        distances = OrderedDict()
        avg_sat = OrderedDict()
        ensemble_sat = OrderedDict()
        centroid_sat = OrderedDict()
        centroid_XL_receptor_copies = OrderedDict()
        print("\nCalculating crosslink statistics:") 
        for l in self.ligands:
            print("ligand ", l)
            out = self._get_XL_statistics(l)
            distances[l] = out[0]
            avg_sat[l] = out[1]
            ensemble_sat[l] = out[2]
            
            centroid_data = self._get_centroid_XL_satisfaction(l)
            centroid_sat[l] = centroid_data[0]
            centroid_XL_receptor_copies[l] = centroid_data[1]
        sat_data = (avg_sat, ensemble_sat, centroid_sat)
        
        # write crosslink satisfaction and distance histograms
        for l in self.ligands:
            self._write_XL_sat(l, sat_data)
            self._write_XL_distance_histogram(l, distances)
        
        # write summary
        self._write_XL_summary(sat_data)

        # plot distance histograms
        hist_files = OrderedDict()
        for l in self.ligands:
            hist_files[l] = os.path.join(self.outdir, 
                            "%s_XL_distance_histogram.txt" % l)
        
        outfn = os.path.join(self.outdir, "XL_distance_histograms.png")
        ligand_colors = {l: self.molinfo[l]["color"] for l in self.ligands}
        plot_2d.plot_all_XL_distance_histogram(hist_files, outfn, ligand_colors,
                                               cutoff=self.cutoff)
        
        # write chimerax scripts for rendering crosslinks mapped to centroid
        # change the molecule name in the key to reflect the correct
        # receptor copy number. This is is necessary for chimerax
        chimerax_centroid_sat = OrderedDict()
        for l in self.ligands:
            this_chimerax_centroid_sat = OrderedDict()
            for k, v in centroid_sat[l].items():
                receptor_name = k[0]
                receptor_copy = centroid_XL_receptor_copies[l][k]
                new_receptor_name = receptor_name + "." + str(receptor_copy)
                new_k = (new_receptor_name, *k[1:])
                this_chimerax_centroid_sat[new_k] = v
            chimerax_centroid_sat[l] = this_chimerax_centroid_sat
                    
        outfn = os.path.join(self.outdir, "XL_maps.cxc")
        plot_chimerax.render_crosslink_maps(self.centroid_pdb_file, 
                        self.receptor, self.ligands,
                        self.molinfo, self.num_receptor_copies,
                        chimerax_centroid_sat, self.outdir, outfn)
        
        
    def _get_XL_distances_frame(self, ligand):
        """
        Get distances of crosslinks from a particular ligand (hence a 
        particular dataset) to the receptor.

        Args:
        ligand (str): Name of ligand molecule.

        Returns:
        (tuple): (distances, copy numbers) of crosslinks to receptor residues.
        Copy numbers are of the receptor monomers for which the crosslink
        distance is a minimum across all receptor monomers / copies. This is
        used for ambiguous crosslinks. 
        """
        
        distances = OrderedDict()

        receptor_copies = OrderedDict()
        for k, (ps_r, p_l) in self.particle_pair_lookup[ligand].items():
            d = [IMP.core.get_distance(p, p_l) for p in ps_r]
            distances[k] = min(d)
            receptor_copies[k] = d.index(distances[k])
        return distances, receptor_copies
    
    def _get_centroid_XL_satisfaction(self, ligand):
        """
        Get distances of crosslinks from a particular ligand (hence a 
        particular dataset) to the receptor, only for the centroid model.

        Args:
        ligand (str): Name of ligand molecule.

        Returns:
        (tuple): (distances, copy numbers) of crosslinks to receptor residues.
        Copy numbers are of the receptor monomers for which the crosslink
        distance is a minimum across all receptor monomers / copies. This is
        used for ambiguous crosslinks. 
        """
        
        self.rmf_handler.load_frame(self.centroid_frame)
        distances, receptor_copies = self._get_XL_distances_frame(ligand)
        sat = OrderedDict()
        for k, v in distances.items():
            sat[k] = float(v <= self.cutoff)
        return sat, receptor_copies
    
    def _get_XL_statistics(self, ligand):
        """
        Calculate XL distances and satisfaction (avg, ensemble, centroid)
        for a given ligand (hence a given crosslink dataset).

        Args:
        ligand (str): Ligand molecule name.

        Returns:
        (tuple): (distances, avg sat, ensemble sat) for each crosslink between
        the receptor and the given ligand.
        """
        
        # result accumulators
        distances = []
        avg_sat = OrderedDict()
        ensemble_sat = OrderedDict()
        for k in self.particle_pair_lookup[ligand]:
            avg_sat[k] = 0.0
            ensemble_sat[k] = 0.0
            
        # frame iteration
        nframes = len(self.frames)
        for i in range(nframes):
            self.rmf_handler.load_frame(self.frames[i])
            this_distances, _  = self._get_XL_distances_frame(ligand)
            distances.extend(list(this_distances.values()))
            for k, v in this_distances.items():
                avg_sat[k] += float(v <= self.cutoff)
        
        # reduce over all frames
        for k in self.particle_pair_lookup[ligand]:
            avg_sat[k] /= float(nframes)
            ensemble_sat[k] = float(avg_sat[k] > 0.0)
        
        return distances, avg_sat, ensemble_sat

    def _write_XL_sat(self, ligand, sat_data):
        """
        Write computed crosslink satisfaction for a given ligand to file.
        
        Args:
        ligand (str): Ligand molecule name.
        
        sat_data (tuple): (Avg. sat, ensemble sat, centroid sat) for 
        each crosslink between the receptor and this ligand.
        """
        
        avg_sat, ensemble_sat, centroid_sat = sat_data
        df = []
        for k in self.particle_pair_lookup[ligand]:
            this_df = (*k,
                       avg_sat[ligand][k],
                       ensemble_sat[ligand][k],
                       centroid_sat[ligand][k])
            df.append(this_df)
        
        colnames = ["receptor", "receptor_residue", "ligand", "ligand_residue",
                    "average_sat", "ensemble_sat", "centroid_sat"]
        df = pd.DataFrame(df, columns=colnames)
        outfn = os.path.join(self.outdir, "%s_XL_satisfaction.csv" % ligand)
        df.to_csv(outfn, index=False, float_format=SAVE_FLOAT_FMT)
    
    def _write_XL_summary(self, sat_data):
        """
        Summarize overall satisfaction metrics for each crosslink dataset
        and write to file.
        
        Args:
        sat_data (tuple): (Avg. sat, ensemble sat, centroid sat) for 
        each crosslink between the receptor and all ligands.
        """
        
        df = []
        avg_sat, ensemble_sat, centroid_sat = sat_data
        for l in self.ligands:
            s1 = np.mean([v for v in avg_sat[l].values()])
            s2 = np.mean([v for v in ensemble_sat[l].values()])
            s3_nsat = np.sum([v for v in centroid_sat[l].values()])
            s3_nxl = len(centroid_sat[l])
            df.append((l, s1, s2, s3_nsat, s3_nxl))
        
        colnames = ["ligand", "avg_sat", "ensemble_sat",
                    "centroid_nsat", "centroid_nXL"]
        df = pd.DataFrame(df, columns=colnames)
        outfn = os.path.join(self.outdir, "summary_XL_satisfaction.csv")
        df.to_csv(outfn, index=False, float_format=SAVE_FLOAT_FMT)
    
    def _write_XL_distance_histogram(self, ligand, distances):
        """
        Bin crosslink distances between receptor and a given ligand, 
        and write the histogram counts to file.
        
        Args:
        ligand (str): Ligand molecule name.
        
        distances (dict): Dict containing array of all possible distances
        for a crosslink dataset across all relevant frames from the given
        cluster of models.
        """
        
        # make distance histogram
        bin_centers, bin_counts = utils.make_histogram(distances[ligand],
                                                       nbins=NBINS)
        
        # write histogram to file
        hist_data = np.array(list(zip(bin_centers, bin_counts)))
        hist_fn = os.path.join(self.outdir, 
                             "%s_XL_distance_histogram.txt" % ligand)
        np.savetxt(hist_fn, hist_data, fmt=SAVE_FLOAT_FMT)
        