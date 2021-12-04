#$ -S /bin/bash
#$ -cwd
#$ -r n
#$ -j y
#$ -R y
#$ -l mem_free=4G
#$ -l h_rt=160:00:00
#$ -l scratch=200G
#$ -o /wynton/scratch/tsanyal
#$ -e /wynton/scratch/tsanyal

hostname
date

# add conda environment
eval "$(conda shell.bash hook)"
conda activate impenv

# IMP environment
IMPENV=$HOME/mysoftware/salilab/imp_release/setup_environment.sh

# get nblib path
# REMEMBER TO ADD NBLIB PATH HERE!

# run
$IMPENV python get_subsampled_models.py -nb $1 -d ./data -m $1 -np 20

hostname
date

qstat -j $JOB_ID
