#!/bin/bash -e

# Set up an environment to run tests under Travis CI (see toplevel .travis.yml)
# or GitHub Actions (see toplevel .github/workflows/build.yml)

IMP_CONDA="imp"

eval "$(conda shell.bash hook)"
conda activate test
conda install --yes -q numpy pandas matplotlib biopython networkx scikit-learn hdbscan tqdm cython setuptools ${IMP_CONDA}
