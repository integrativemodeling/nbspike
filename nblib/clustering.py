"""
Interface based clustering of docked nanobody models.
"""

import os
import tqdm
import itertools
import pandas as pd
import numpy as np
import networkx as nx
from collections import OrderedDict

from sklearn.cluster import DBSCAN

import IMP
import IMP.atom
import IMP.core
import IMP.rmf
import RMF

from . import epitopelib, utils
from .graphics import RigidPDBWriter, plot_chimerax

# pyRMSD calculator specs (no alignment needed)
CALCTYPE = "NOSUP_OMP_CALCULATOR"

# float format for writing to file.
SAVE_FLOAT_FMT = "%3.3f"

# how many models to keep for chimerax viewing
NUM_TOP_MODELS = 10


class InterfaceCluster:
    """
    Interface based clustering of docked nanobody models.
    """
    
    def __init__(self, rmf_file, topology_file, 
                 pdb_dir=".", fasta_dir=".",
                 read_freq=1, num_omp_threads=1, outdir="."):
        """
        Constructor.
        
        Args:
        rmf_file (str): RMF file name.
        
        topology_file (str): PMI topology file.
        
        pdb_dir (str, optional): Directory containing all pdb files used in modeling and mentioned in the topology file. Defaults to ".".
        
        fasta_dir (str, optional): Directory containing all FASTA files used in modeling and mentioned in the topology file. Defaults to ".".
        
        read_freq (int, optional): Read every <read_freq>th frames only.
        Defaults to 1.
        
        num_omp_threads (int, optional): Number of threads to run
        parallel computations. Defaults to 1.
        
        outdir (str, optional): Top level output directory for all clustering
        results. Defaults to ".".
        """
        
        # set the number of OMP threads
        self.num_omp_threads = np.int32(num_omp_threads)
         
        # parse the topology file
        th = utils.TopologyHandler(topology_file, pdb_dir=pdb_dir,
                                   fasta_dir=fasta_dir)
        self.receptor, self.ligands = th.receptor, th.ligands
        self.molinfo = th.molinfo
        self.num_receptor_copies = np.int32(th.num_receptor_copies)
        
        # create the output dir
        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir, exist_ok=True)
        
        # parse RMF file
        self.read_freq = read_freq
        self.rmf_handler = utils.RMFHandler(rmf_file, self.read_freq)
        self.frames = self.rmf_handler.frames
        
        # pdb writer
        self.pdb_writer = RigidPDBWriter(rmf_file=rmf_file,
                                         topology_file=topology_file,
                                         pdb_dir=pdb_dir,
                                         fasta_dir=fasta_dir,
                                         read_freq=read_freq)
     
           
    # -----------------------------
    # WORKSHORSE METHODS (i.e. API)
    # -----------------------------
    def calculate_interface_distance_matrix(self, epitope_cutoff=8.0):
        """
        Calculate all-by-all distance matrix for receptor-(all) ligand-residue
        epitope pairs, based on interface similarity metrics. Currently, the FCC metric is used. See:
        
        "Clustering biomolecular complexes by residue contacts similarity"
        Rodrigues, Trellet, Schimtz, Kastritis, Karaca, Melquinod & Bonvin
        Proteins: Structure, Function and Bioinformatics, 2012

        Args:
        epitope_cutoff (float, optional): Max. distance between a receptor
        and ligand residue to be qualified as an epitope pair. Defaults to 8.0.
        """
        
        # get particles
        ps, nps = self._get_particles()
        # save the particle number list as a class attribute
        self.n_particles = np.array(nps, dtype=np.int32)
        
        # get radii
        radii = np.array([p.get_radius() for p in ps],
                         dtype=np.float32)
        
        # get coordinates
        coords_fn = os.path.join(self.outdir, "coords.npy")
        if not os.path.isfile(coords_fn):
            print("Extracting model coordinates...")
            self.coords = self._get_coords(ps)
            np.save(coords_fn, self.coords)
        else:
            print("Loading saved coordinates")
            self.coords = np.load(coords_fn)
            
        # get epitopes
        epitope_cutoff = np.float32(epitope_cutoff)
        epitopes_fn = os.path.join(self.outdir, 
                                   "epitopes_%2.2f.npy" % epitope_cutoff)
        if not os.path.isfile(epitopes_fn):
            print("\nExtracting epitopes...")
            self.epitopes = epitopelib.get_epitope_pair(self.coords, radii,
                                          epitope_cutoff, self.n_particles,
                                          self.num_receptor_copies,
                                          self.num_omp_threads)
    
            np.save(epitopes_fn, self.epitopes)
        else:
            print("\nLoading saved epitopes")
            self.epitopes = np.load(epitopes_fn)
    
        # get interface distance matrix
        distmat_fn = os.path.join(self.outdir,
                                  "distmat_%2.2f.npy" % epitope_cutoff)
        if not os.path.isfile(distmat_fn):
            print("\nCalculating distance matrix...")
            self.distmat = epitopelib.get_interface_distance_matrix(
                self.epitopes, self.num_omp_threads)
            np.save(distmat_fn, self.distmat)
        else:
            print("\nLoading distance matrix from file")
            self.distmat = np.load(distmat_fn)
    
    
    def cluster(self, cluster_cutoff=0.3, min_cluster_size=10):
        """
        Cluster docked models and write cluster statistics. Also render
        top models and centroids from each cluster. Assumes that the distance
        matrix has already been constructed.

        Args:
        cluster_cutoff (float, optional): Interface distance threshold
        for two models to be assigned to different clusters. Defaults to 0.3.
        
        min_cluster_size (int, optional): Min. size of a cluster.
        Defaults to 10.
        """
        
        self.clusters = self._cluster_Taylor_Butina(cluster_cutoff,
                                                    min_cluster_size)
        
        if not self.clusters:
            print("No clusters found.")
            return
        
        # write cluster sizes
        print("Extracted %d clusters" % len(self.clusters))
        nframes = len(self.frames)
        cluster_sizes =[(len(c), len(c) / float(nframes)) 
                        for c in self.clusters]
        cluster_sizes =  np.array(cluster_sizes)
        cluster_size_fn = os.path.join(self.outdir, "cluster_size.txt")
        np.savetxt(cluster_size_fn, cluster_sizes, fmt=SAVE_FLOAT_FMT)

        # create dirs for each cluster
        self.cluster_dirs = []
        for i in range(len(self.clusters)):
            d = os.path.join(self.outdir, "cluster." + str(i))
            os.makedirs(d, exist_ok=True)
            self.cluster_dirs.append(d)
        
        # write info for each cluster
        print("\nWriting structures for:")
        for i, c in enumerate(self.clusters):
            print("cluster ", i)
            # write all rmf frames in the cluster (centroid first)
            cluster_fn = os.path.join(self.cluster_dirs[i], "frames.txt")
            cluster_frames = [x*self.read_freq for x in c]
            np.savetxt(cluster_fn, cluster_frames, fmt="%d")    
            
            # write centroid frame as a rmf
            self._write_centroid_rmf(i)
            
            # write centroid frame as a pdb file
            self._write_centroid_pdb(i)
            
            # write the top ten models as a pdb file
            self._write_top_models_pdb(i, NUM_TOP_MODELS)
        
    
    def calculate_precision(self):
        """
        Calculate overall and per-ligand precision metrics for each cluster.
        """
        
        if not self.clusters:
            return

        print("\nCalculating overall precision for each cluster...")
        cluster_precisions = self.get_interface_distance()
        cluster_precisions = np.array(cluster_precisions)
        cluster_precision_fn = os.path.join(self.outdir, 
                                            "cluster_precision.txt")
        np.savetxt(cluster_precision_fn, cluster_precisions,
                   fmt=SAVE_FLOAT_FMT)
    
        # calculate precision metrics for each ligand
        print("\nCalculating metrics for each ligand in:")
        for i, c in enumerate(self.clusters): 
            print("> cluster ", i)
            # ligand RMSD
            lrmsds = []
            for l in self.ligands:
                rmsd = self.get_ligand_RMSD(i, l)
                lrmsds.append(rmsd)
        
            # interface ligand RMSD
            ilrmsds = []
            for l in self.ligands:
                rmsd = self.get_interface_ligand_RMSD(i, l)
                ilrmsds.append(rmsd)
        
            # interface distance
            idists = []
            for l in self.ligands:
                d = self.get_interface_ligand_distance(i, l)
                idists.append(d)
        
            # compile into an overall data structure and write to file
            df = list(zip(self.ligands, lrmsds, ilrmsds, idists))
            df = pd.DataFrame(df, columns=["ligand", "L-RMSD",
                                           "iL-RMSD", "i-Distance"])
        
            ligand_metrics_fn = os.path.join(self.cluster_dirs[i],
                                            "ligand_metrics.csv")
            df.to_csv(ligand_metrics_fn, index=False,
                      float_format=SAVE_FLOAT_FMT)    
     
    
    # -------------
    # PREPROCESSING
    # -------------
    def _get_particles(self):
        """
        Get particles for receptor and ligands from RMF file.
        
        Returns:
        (tuple): (list of spherical, i.e. IMP.core.XYZR decorated particles,
        list of particle numbers). Each element of the list is the number
        of particles of a molecule (chain). All receptor chains come first,
        followed by ligands, in the same order used in the topology file.
        """
        
        ps = []
        nps = []

        # receptor
        for i in range(self.num_receptor_copies):
            this_ps = self.rmf_handler.get_particles(self.receptor,
                                                    copy_number=i,
                                                    has_radius=True)
            ps.extend(this_ps)
            nps.append(len(this_ps))

        # ligands
        for l in self.ligands:
            this_ps = self.rmf_handler.get_particles(l, has_radius=True)
            nps.append(len(this_ps))
            ps.extend(this_ps)
        return ps, nps
    
    def _get_coords(self, ps):
        """
        Get coordinates of particles over all RMF frames.
        
        Args:
        ps (list): Spherical IMP particles.

        Returns:
        (3D single precision (np.float32) array): (X,Y,Z) coordinates
        of all particles over all frames.
        """
        
        nframes = len(self.frames)
        coords = []
        for i in tqdm.trange(nframes):
            self.rmf_handler.load_frame(self.frames[i])
            this_coords = [np.array(p.get_coordinates()) for p in ps]
            coords.append(this_coords)
        coords = np.array(coords, dtype=np.float32)
        return coords

    
    # ----------
    # CLUSTERING
    # ----------
    def _cluster_Taylor_Butina(self, cluster_cutoff, min_cluster_size):
        """
        Implements the Taylor-Butina asymmetric clustering algorithm
        for the FCC interface metric. Assumes that the distance matrix
        has already been computed.
        
        See:
        "Clustering biomolecular complexes by residue contacts similarity"
        Rodrigues, Trellet, Schimtz, Kastritis, Karaca, Melquinod & Bonvin
        Proteins: Structure, Function and Bioinformatics, 2012
        
        Also see:
        "Incorporating sequential information into traditional classification models by using an element/position-sensitive SAM"
        Prinzie, Van den Poel
        Decision Support Systems, 2006
        
        Args:
        cluster_cutoff (float, optional): Interface distance threshold
        for two models to be assigned to different clusters.
        
        min_cluster_size (int, optional): Min. size of a cluster.

        Returns:
        (list of lists):  The i^th list is a collection of the RMF frames 
        that make up the i^th cluster, with the first frame being the centroid
        for that cluster. These cluster lists are arranged in decreasing order
        of cluster size.
        """
        
        print("\nTaylor Butina clustering...")
        # create neighbor lists in the form of a directed graph
        nodes = list(range(self.distmat.shape[0]))
        edges = [(i, j) for (i, j) in itertools.permutations(nodes, 2)
             if self.distmat[i,j] <= cluster_cutoff]
    
        # create nearest neighbor table as a graph
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
    
        # remove true singletons
        singleton_nodes = [u for u in G.nodes if G.out_degree(u) == 0]
        G.remove_nodes_from(singleton_nodes)
        print("> removing %d true singletons from %d models" % \
              (len(singleton_nodes), len(nodes)))
    
        # main-loop
        c_iter = 0
        clusters = []
        degrees= sorted(G.out_degree, key=lambda x: x[1], reverse=True)
        while (1):
            # get node with largest neighbor list
            degrees= sorted(G.out_degree, key=lambda x: x[1], reverse=True)
            
            # check
            is_empty = len(G) == 0
            is_neigh_lean = all([x[1] < min_cluster_size for x in degrees])
            if is_empty or is_neigh_lean:
                break
            
            # form cluster
            print("> cluster iteration %d" % c_iter)
            centroid = degrees[0][0]
            members = list(G.neighbors(centroid))
            this_cluster = [centroid] + sorted(members)
            clusters.append(this_cluster)

            # update graph
            G.remove_nodes_from(this_cluster)
            c_iter += 1
            
        return clusters
    
    def _cluster_DBSCAN(self, cluster_cutoff, min_cluster_size):
        """
        Wrapper over the DBSCAN clustering algorithm provided by 
        scikit-learn.
        
        Args:
        cluster_cutoff (float, optional): Interface distance threshold
        for two models to be assigned to different clusters.
        
        min_cluster_size (int, optional): Min. size of a cluster.

        Returns:
        (list of lists):  The i^th list is a collection of the RMF frames 
        that make up the i^th cluster. No centroids are computed, since
        DBSCAN is a centroid-free method.
        """
        
        cl = DBSCAN(eps=cluster_cutoff, min_samples=min_cluster_size,
                    metric="precomputed").fit(self.distmat)
        unique_labels = set([l for l in cl.labels_ if l != -1])
        cluster_dict = {x: [] for x in unique_labels}
        for i, l in enumerate(cl.labels_):
            if l != -1: cluster_dict[l].append(i)
        clusters = list(cluster_dict.values())
        return clusters
    

    # -------------
    # STRUCTURE I/O
    # -------------
    def _write_centroid_rmf(self, cluster_id):
        """
        Write the centroid of a given cluster to a RMF file.
        
        Adapted from imp-sampcon:
        https://github.com/salilab/imp-sampcon/blob/a4813ef8a90baa585d92ae8f4f478ab762864b1f/pyext/src/exhaust.py#L104
        
        Args:
        cluster_id (int): Id of a cluster. Cluster with increasing id have
        lower size.
        """
        
        idx = self.clusters[cluster_id][0]
        centroid_model_id = self.frames[idx]
        
        self.rmf_handler.update()
        in_rh = self.rmf_handler.file_handle
        
        outfn = os.path.join(self.cluster_dirs[cluster_id],
                             "cluster_center_model.rmf3")
        out_rh = RMF.create_rmf_file(outfn)
        
        RMF.clone_file_info(in_rh, out_rh)
        RMF.clone_hierarchy(in_rh, out_rh)
        RMF.clone_static_frame(in_rh, out_rh)
        in_rh.set_current_frame(RMF.FrameID(centroid_model_id))
        out_rh.add_frame("f0")
        RMF.clone_loaded_frame(in_rh, out_rh)
    
    def _write_centroid_pdb(self, cluster_id):
        """
        Write the centroid of given cluster to a PDB file.

        Args:
        cluster_id (int): Id of a cluster. Cluster with increasing id have
        lower size.
        """
        centroid_frame = self.clusters[cluster_id][0] * self.read_freq
        self.pdb_writer.set_frames(centroid_frame)
        out_pdb_fn = os.path.join(self.cluster_dirs[cluster_id], 
                                  "cluster_center_model.pdb")
        self.pdb_writer.write(out_pdb_fn)
    
    def _write_top_models_pdb(self, cluster_id, k=1):
        """
        Write the top k models from a given cluster to a single PDB file.
        
        Args:
        cluster_id (int): Id of a cluster. Cluster with increasing id have
        lower size.
            
        k (int, optional): Number of models to write. Defaults to 1.
        """
        
        # get centroid and members of this cluster
        centroid = self.clusters[cluster_id][0]
        members = self.clusters[cluster_id][1:]
        if len(members) < k:
            k = len(members)
        
        # get k models closest to the centroid
        idist = np.array([self.distmat[centroid, i] for i in members])
        idx = np.argsort(idist)[:k]
        top_members = [m for m in np.array(members, dtype=int)[idx]]      
        top_members = [centroid] + top_members
        
        # write frames
        frames = [m*self.read_freq for m in top_members]
        self.pdb_writer.set_frames(frames)
        out_pdb_fn = os.path.join(self.cluster_dirs[cluster_id],
                                  "top_%d_models.pdb" % k)
        self.pdb_writer.write(out_pdb_fn)
        
        # write chimerax script
        out_cxc_fn = os.path.join(self.cluster_dirs[cluster_id],
                                  "top_%d_models.cxc" % k)            
        plot_chimerax.render_cluster_variability(pdb_file=out_pdb_fn,
                                receptor=self.receptor,
                                ligands=self.ligands,
                                molinfo=self.molinfo,
                                num_receptor_copies=self.num_receptor_copies,
                                outfile=out_cxc_fn)
        
        
    # -----------------
    # PRECISION METRICS
    # -----------------    
    def get_interface_distance(self):
        cluster_precisions = []
        for c in self.clusters:
            e = self.epitopes[c, :]
            p = epitopelib.get_average_interface_distance(e, 
                                        self.num_omp_threads)
            cluster_precisions.append(p)
        return cluster_precisions
    
    
    def get_interface_ligand_distance(self, cluster_id, ligand):
        """
        For a specific ligand, calculate avg. interface distance (i.e. 1.0 - fractional similarity) from the cluster centroid, for all cluster members.

        Args:
        cluster_id (int): Cluster id.
        
        ligand (str): Name of ligand molecule as mentioned in topology file.

        Returns:
        (float): Avg. interface distance from cluster centroid for that ligand.
        """
        
        nr = sum(self.n_particles[:self.num_receptor_copies])
        nl_all = sum(self.n_particles) - nr
        
        idx = self.ligands.index(ligand)
        nl = self.n_particles[self.num_receptor_copies+idx]
        nps_ligands = self.n_particles[self.num_receptor_copies:]
        l_start = sum(nps_ligands[:idx])
        l_stop = l_start + nl
        
        l_indices = []
        for i in range(nr):
            this_l_indices = list(range(i*nl_all+l_start, i*nl_all+l_stop))
            l_indices.extend(this_l_indices)
            
        l_epitopes = self.epitopes[:, l_indices]
        l_epitopes_cluster = l_epitopes[self.clusters[cluster_id], :]
        out = epitopelib.get_average_interface_distance(l_epitopes_cluster,
                                                     self.num_omp_threads)
        return out
    
    
    def get_ligand_RMSD(self, cluster_id, ligand):
        """
        Calculate avg RMSD of a given ligand molecule for all models in
        a cluster, using the cluster centroid as a reference model.
        
        Args:
        cluster_id (int): Cluster id.
        
        ligand (str): Name of ligand molecule as mentioned in topology file.

        Returns:
        (float): Avg. ligand RMSD from cluster centroid.
        """

        from pyRMSD.RMSDCalculator import RMSDCalculator

        # get coords
        idx = self.ligands.index(ligand)
        start = sum(self.n_particles[:(self.num_receptor_copies+idx)])
        stop = start + self.n_particles[self.num_receptor_copies+idx]
        coords = self.coords[self.clusters[cluster_id], start:stop, :]
        # coords have to be cast to double precision else pyRMSD yells at you
        coords = np.array(coords, np.float64)
        
        # get number of models for normalizing
        n_models = len(self.clusters[cluster_id]) - 1
        
        # calculate rmsds
        calc = RMSDCalculator(CALCTYPE, coords)
        calc.setNumberOfOpenMPThreads(self.num_omp_threads)
        rmsds = calc.oneVsTheOthers(0)
        return np.sum(rmsds) / float(n_models)
    
    
    def get_interface_ligand_RMSD(self, cluster_id, ligand):
        """
        Calculate avg RMSD of only the epitope residues of a given 
        ligand molecule for all models in a cluster, 
        using the cluster centroid as a reference model.
        
        Args:
        cluster_id (int): Cluster id.
        
        ligand (str): Name of ligand molecule as mentioned in topology file.

        Returns:
        (float): Avg. interface ligand RMSD from cluster centroid.
        """

        from pyRMSD.RMSDCalculator import RMSDCalculator    
        
        # get epitope for centroid frame
        nr = sum(self.n_particles[:self.num_receptor_copies])
        centroid_frame = self.clusters[cluster_id][0]
        e = self.epitopes[centroid_frame].reshape(nr, -1)
        
        # extract the epitope corresponding to the given ligand
        idx = self.ligands.index(ligand)
        start = sum(self.n_particles[:(self.num_receptor_copies+idx)]) - nr
        stop = start + self.n_particles[self.num_receptor_copies+idx]
        centroid_ligand_epitope = e[:, start:stop]
        
        # get ligand epitope residue indexes
        # relative indexes
        iresindexes_ = set(np.where(centroid_ligand_epitope == 1)[-1])
        # absolute indexes
        iresindexes = [start + nr + i for i in iresindexes_]
        
        # get coords from residue indexes from all frames in the cluster: 
        # these are interface residues of ligand corresponding to 
        # the centroid frame, and will be used as the definition of 
        # the "ligand interface"
        all_coords = self.coords[self.clusters[cluster_id]]
        coords = np.array(all_coords[:, iresindexes], np.float64)
        
        # get number of models for normalizing
        n_models = len(self.clusters[cluster_id]) - 1
        
        # calculate rmds for interface_ligand_coords
        calc = RMSDCalculator(CALCTYPE, coords)
        calc.setNumberOfOpenMPThreads(self.num_omp_threads)
        rmsds = calc.oneVsTheOthers(0)
        return np.sum(rmsds) / float(n_models)
    
    
