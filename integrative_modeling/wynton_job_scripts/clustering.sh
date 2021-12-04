#$ -S /bin/bash
#$ -cwd
#$ -r n
#$ -j y
#$ -R y
#$ -l mem_free=8G
#$ -l h_rt=160:00:00
#$ -l scratch=200G
#$ -t 1-2
#$ -o /wynton/scratch/tsanyal
#$ -e /wynton/scratch/tsanyal

hostname
date

# add conda environment
eval "$(conda shell.bash hook)"
conda activate impenv

# IMP environment
IMPENV=$HOME/mysoftware/salilab/imp_release/setup_environment.sh

# REMEMBER TO ADD NBLIB PATH HERE!

# analysis directory for this subsample of models
i=$(expr $SGE_TASK_ID)
let i-=1

RMF_FILE=$1/subsampled_models/models_$i.rmf3
OUTDIR=$1/analysis/analysis_$i/clusters

# run clustering
$IMPENV python clustering.py -nb $1 -r $RMF_FILE -d ./data -o $OUTDIR -n 20

hostname
date

qstat -j $JOB_ID
