#$ -S /bin/bash
#$ -cwd
#$ -r n
#$ -j y
#$ -R y
#$ -l mem_free=4G
#$ -l h_rt=160:00:00
#$ -l scratch=200G
#$ -pe smp 8
#$ -t 1-20
#$ -o /wynton/scratch/tsanyal
#$ -e /wynton/scratch/tsanyal

hostname
date

export JOB_NAME=$1

# add conda environment
eval "$(conda shell.bash hook)"
conda activate impenv

# IMP environment
IMPENV=$HOME/mysoftware/salilab/imp_release/setup_environment.sh

# run directory for this independent run
i=$(expr $SGE_TASK_ID)
RUNDIR=./$1/run_$i

# run sampling for this run
if [ ! -d $RUNDIR ]; then
    mkdir -p $RUNDIR 
    cd $RUNDIR   
    # REMEMBER TO ADD NBLIB PATH HERE!
    $IMPENV mpirun -np $NSLOTS python ../../modeling.py -nb $1 -d ../../data
    cd .. 
fi

hostname
date

qstat -j $JOB_ID
