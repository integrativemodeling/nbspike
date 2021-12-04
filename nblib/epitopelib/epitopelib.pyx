# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: extra_compile_args=[-fopenmp, -O3]
# distutils: extra_link_args=-fopenmp

"""
Collection of Cython functions for calculating receptor-ligand interface
residues, and interface similarity measures. This library works specifically
with single precision floats (float32 in numpy parlance) and python floats 
(which are double precision) must be cast to single precision before passing
into arguments of functions below.
"""

import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libc.math cimport fabs, sqrt
from cython.parallel import prange

ctypedef np.float32_t DTYPE_t
DTYPE = np.float32

ctypedef np.int32_t ITYPE_t
ITYPE = np.int32

ctypedef np.uint8_t BTYPE_t
BTYPE = np.uint8

cdef DTYPE_t EPS = 1e-10


# ---------------------
# EPITOPE DETERMINATION
# ---------------------
cdef inline BTYPE_t is_epitope_pair(DTYPE_t [:] coord_r, DTYPE_t radius_r,
                                    DTYPE_t [:] coord_l, DTYPE_t radius_l, 
                                    DTYPE_t cutoff) nogil:
    """
    C-only function that checks if the given receptor and ligand residues
    are interfacially proxmial.

    Args:
    coord_r (1D array of numpy.float32s): (X,Y,Z) coords of 
    receptor residue.
    
    radius_r (numpy.float32): Radius of receptor residue sphere.

    coord_l (1D array of numpy.float32s): (X,Y,Z) coords of 
    ligand residue.
    
    radius_l (numpy.float32): Radius of ligand residue sphere.

    cutoff (numpy.float32): Max. receptor-ligand distance for the pair to 
    qualify as an epitope.

    Returns:
    (np.uint8): 1 if the given receptor-ligand-residue-pair is interfacial
    and 0 otherwise.
    """

    cdef DTYPE_t dx, dy, dz, r
    
    dx = coord_r[0] - coord_l[0]
    dy = coord_r[1] - coord_l[1]
    dz = coord_r[2] - coord_l[2]
    r = sqrt(dx*dx + dy*dy + dz*dz)
    #r -= (radius_r + radius_l)

    if r <= cutoff:
        return 1
    else:
        return 0


cpdef np.ndarray[BTYPE_t, ndim=2] get_epitope_pair(DTYPE_t [:, :, ::1] coords, 
                                                   DTYPE_t [:] radii,
                                                   DTYPE_t cutoff,
                                                   ITYPE_t [:] n_particles,
                                                   ITYPE_t n_receptor_copies,
                                                   ITYPE_t n_threads):
    """
    Get all epitope pairs for the receptor and ligand.

    Args:
    coords (3D array of numpy.float32s): (X,Y,Z) coords of all receptor
    residues followed by all ligand residues, for each model,
    stacked into a single array.
    
    radius_r (1D array of numpy.float32s): Radii of all receptor residues
    followed by radii of all ligand residues. 

    cutoff (numpy.float32): Max receptor-ligand distance for the pair to 
    qualify as an epitope.

    n_particles (1D array of numpy.int32s): List of number of particles 
    for each receptor chain, followed by that for each ligand.

    n_receptor_copies(numpy.int32): Number of copies of the receptor chain.
    This is a required argument even if it is 1.

    n_threads (numpy.int32): Number of threads to use for parallel computation.

    Returns:
    (2D array of np.uint8s): For NR receptor residues, NL ligand residues
    and M models, returns a [M, (NR X NL)] 2D binary indicator array, 
    where array[i, j] is a 1 or 0 according to whether the 
    j^th receptor-ligand residue pair in the i^th model is an epitope pair 
    or not.
    """

    cdef Py_ssize_t n_frames, nr, nl, ii, i, j
    n_frames = coords.shape[0]
    nr = sum(n_particles[:n_receptor_copies])
    nl = sum(n_particles) - nr

    cdef DTYPE_t [:, :, ::1] coords_r = coords[:, 0:nr, :]
    cdef DTYPE_t [:] radii_r = radii[0:nr]
    
    cdef DTYPE_t [:, :, ::1] coords_l = coords[:, nr:(nr+nl), :]
    cdef DTYPE_t [:] radii_l = radii[nr:(nr+nl)]
    
    epitope_pair = np.zeros([n_frames, nr*nl], BTYPE)
    cdef BTYPE_t [:, ::1] epitope_pair_ = epitope_pair

    for ii in prange(n_frames, num_threads=n_threads, schedule="static",
                     nogil=True):
        for i in range(nr):
            for j in range(nl):
                epitope_pair_[ii, i*nl+j] = is_epitope_pair(coords_r[ii, i],
                                                            radii_r[i],
                                                            coords_l[ii, j], radii_l[j],
                                                            cutoff)
    return epitope_pair


# ----------
# FCC METRIC
# ----------

cdef inline DTYPE_t fcc(BTYPE_t [:] epitope_A,
                        BTYPE_t [:] epitope_B) nogil:
    """
    C-only function that calculates the "(F)raction of (C)ommon (C)ontacts"
    of FCC, between two alternate models of the receptor-ligand interface.
    
    Taken from:
    "Clustering biomolecular complexes by residue contacts similarity"
    Rodrigues, Trellet, Schimtz, Kastritis, Karaca, Melquinod & Bonvin
    Proteins: Structure, Function and Bioinformatics, 2012
    
    Args:
    epitope_A (1D array of numpy.int32s): Binary indicator array of whether
    each receptor-ligand-receptor pair is an epitope or not, in model A.  
    
    epitope_B (1D array of numpy.int32s): Binary indicator array of whether
    each receptor-ligand-receptor pair is an epitope or not, in model B.

    Returns:
    (numpy.float32) Fraction of common contacts between receptor and ligand
    in models A and B. Note this is an asymmetric measure. FCC(A,B) is not
    equal to FCC(B, A).
    """

    cdef:
        Py_ssize_t n_particles, i
        DTYPE_t c_A, c_AB
        
    n_particles = epitope_A.shape[0]
    c_A = 0.0
    c_AB = 0.0

    for i in range(n_particles):
        c_A += <DTYPE_t>epitope_A[i]
        c_AB += <DTYPE_t>(epitope_A[i] * epitope_B[i])

    return c_AB / (c_A + EPS)


cdef inline DTYPE_t jaccard(BTYPE_t [:] epitope_A,
                            BTYPE_t [:] epitope_B) nogil:
    """
    C-only function that calculates the Jaccard distance between
    two alternate models of the receptor-ligand interface.
    
    Args:
    epitope_A (1D array of numpy.int32s): Binary indicator array of whether
    each receptor-ligand-receptor pair is an epitope or not, in model A.  
    
    epitope_B (1D array of numpy.int32s): Binary indicator array of whether
    each receptor-ligand-receptor pair is an epitope or not, in model B.

    Returns:
    (numpy.float32) Jaccard distance between receptor and ligand interfaces
    in models A and B.
    """
    
    cdef:
        Py_ssize_t n_particles, i
        DTYPE_t c_A, c_B, c_AB
   
    n_particles = epitope_A.shape[0]
    c_A = 0.0
    c_B = 0.0
    c_AB = 0.0

    for i in range(n_particles):
        c_A += <DTYPE_t>epitope_A[i]
        c_B += <DTYPE_t>epitope_B[i]
        c_AB += <DTYPE_t>(epitope_A[i] * epitope_B[i])
    
    return c_AB / (c_A + c_B - c_AB + EPS)


# -------------------------
# INTERFACE DISTANCE MATRIX
# -------------------------
cpdef np.ndarray[DTYPE_t, ndim=2] get_interface_distance_matrix(
    BTYPE_t [:, ::1] epitopes,
    ITYPE_t n_threads):

    """
    Calculate an all-by-all distance matrix between receptor-ligand-interface
    pairs across a number of alternate models.

    Args:
    epitopes (2D array of numpy.uint8s): Binary indicators for each
    receptor-ligand pair, in each model.
    
    n_threads (numpy.int32): Number of threads to use for parallel computation.

    Returns:
    (2D array of numpy.float32s): Distance matrix where D[i,j] is the 
    distance, i.e., 1.0 - fractional similarity, between the receptor-ligand
    interfaces in docked complex models i and j. Currently, only FCC is used
    as an interface similarity metric.
    """

    cdef Py_ssize_t n_frames, n_particles, i, j

    n_frames = epitopes.shape[0]
    n_particles = epitopes.shape[1]

    mat = np.zeros([n_frames, n_frames], dtype=DTYPE)
    cdef DTYPE_t [:, ::1] mat_ = mat 

    for i in prange(n_frames, num_threads=n_threads, schedule="guided",
                    nogil=True):
        for j in range(n_frames):
            if i == j: continue
            mat_[i, j] = 1.0 - fcc(epitopes[i,:], epitopes[j,:])
    return mat


cpdef DTYPE_t get_average_interface_distance(BTYPE_t [:, ::1] epitopes,
                                             ITYPE_t n_threads):
    """
    Calculate the average interface distance of a given set of models from the
    first model. Thus the first model is used as some sort of a reference.
    For instance, the first model may be a cluster centroid, and the other
    models may be members of that cluster.

    Args:
    epitopes (2D array of numpy.uint8s): Binary indicators for each
    receptor-ligand pair, in each model. Model number 0 is the reference.

    n_threads (numpy.int32): Number of threads to use for parallel computation.

    Returns:
    (numpy.float32): Avg. interfacial distance, i.e., 
    1.0 - fractional similarity, between the receptor-ligand
    interfaces of all models and the first model.
    """
    
    cdef:
        Py_ssize_t i, n_frames
        DTYPE_t dist
    
    n_frames = epitopes.shape[0]
    dist = 0.0

    for i in prange(1, n_frames, num_threads=n_threads, nogil=True):
        dist += fcc(epitopes[0,:], epitopes[i,:])
    return 1.0 - (dist / <DTYPE_t>n_frames)
