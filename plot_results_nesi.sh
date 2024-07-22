#!/bin/bash -e

#SBATCH --job-name          plot_results_psp
#SBATCH --mem               15G
#SBATCH --time              01:30:00
#SBATCH --mail-type         BEGIN,END,FAIL
#SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a

source venv/bin/activate

echo "JOB STARTED"
date

python plot_results.py
#python plot_results_wind.py

echo "FINISHED"
