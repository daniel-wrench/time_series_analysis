#!/bin/bash -e

#SBATCH --job-name          get_gapped_sf

#SBATCH --partition         quicktest
##SBATCH --reservation	    SpjReservation
##SBATCH --nodelist          spj01

#SBATCH --array             0-9
#SBATCH --cpus-per-task     1
#SBATCH --mem               2G
#SBATCH --time              0:10:00
#SBATCH --output            %x.out
#SBATCH --error             %x.err
#SBATCH --mail-type         BEGIN,END,FAIL
#SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load GCC/11.3.0
module load Python/3.10.4

source venv/bin/activate

echo "JOB STARTED"
date

index=$SLURM_ARRAY_TASK_ID

# Load the JSON file corresponding to the files for this core
input_file_list="input_file_lists/input_files_core_${index}.json"

times_to_gap = 5

#rm -r data/processed/*
python get_gapped_sf.py $index $input_file_list $times_to_gap

echo "FINISHED"
