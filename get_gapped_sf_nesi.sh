#!/bin/bash -e

#SBATCH --job-name          get_gapped_sf
#SBATCH --array             0-434
#SBATCH --mem               2G
#SBATCH --time              01:00:00
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a

source venv/bin/activate

echo "JOB STARTED"
date

index=$SLURM_ARRAY_TASK_ID

# Load the JSON file corresponding to the files for this core
input_file_list="input_file_lists/input_files_core_${index}.json"
output_dir="/nesi/project/vuw04187/data/processed/"
times_to_gap=10

rm -r data/processed/*.pkl
python get_gapped_sf.py $index $input_file_list $times_to_gap $output_dir

echo "FINISHED"
