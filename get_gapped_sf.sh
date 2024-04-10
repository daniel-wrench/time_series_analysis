#!/bin/bash -e

#SBATCH --job-name          get_gapped_sf
#SBATCH --partition         quicktest
#SBATCH --nodelist          amd01n01
#SBATCH --mem               10G
#SBATCH --cpus-per-task     12
#SBATCH --time              00:10:00
#SBATCH --output            %x.out
#SBATCH --error             %x.err
##SBATCH --mail-type         ALL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load GCC/11.3.0
module load OpenMPI/4.1.4
module load Python/3.10.4

source venv/bin/activate

echo "JOB STARTED"
date

mpirun --oversubscribe -n 12 python get_gapped_sf.py

echo "FINISHED"
