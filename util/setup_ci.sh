#!/bin/bash -e

# Set up an environment to run tests under Travis CI (see toplevel .travis.yml)
# or GitHub Actions (see toplevel .github/workflows/build.yml)

if [ $# -ne 1 ]; then
  echo "Usage: $0 python_version"
  exit 1
fi

python_version=$1

# get conda-forge, not main, packages
conda config --remove channels defaults || :
conda config --add channels conda-forge
IMP_CONDA="imp"

conda create --yes -q -n python${python_version} -c salilab python=${python_version} numpy pandas matplotlib biopython networkx scikit-learn hdbscan tqdm cython ${IMP_CONDA}
eval "$(conda shell.bash hook)"
conda activate python${python_version}
