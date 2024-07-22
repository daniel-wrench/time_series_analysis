#!/bin/bash -e

#SBATCH --job-name          standardise_wind
#SBATCH --mem               2G
#SBATCH --time              0:30:00
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a

source venv/bin/activate

echo "JOB STARTED"
date

python standardise_wind.py

echo "FINISHED"
