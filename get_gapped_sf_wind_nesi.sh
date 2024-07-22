#!/bin/bash -e

#SBATCH --job-name          get_gapped_sf_wind
#SBATCH --mem               10G
#SBATCH --time              0:30:00
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a

source venv/bin/activate

echo "JOB STARTED"
date

times_to_gap=5

rm -r data/processed/external/*.pkl
python get_gapped_sf_wind.py $times_to_gap

echo "FINISHED"
