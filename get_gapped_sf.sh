#!/bin/bash -e

#SBATCH --job-name          get_gapped_sf
#SBATCH --partition         parallel
##SBATCH --reservation	    SpjReservation
#SBATCH --nodelist          amd02n04
## spj01
#SBATCH --mem               40G
#SBATCH --cpus-per-task     20
#SBATCH --time              00:30:00
#SBATCH --output            %x.out
#SBATCH --error             %x.err
#SBATCH --mail-type         ALL
#SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load GCC/11.3.0
module load OpenMPI/4.1.4
module load Python/3.10.4

source venv/bin/activate

echo "JOB STARTED"
date

mpirun --oversubscribe -mca pml ucx -mca btl '^uct,ofi' -mca mtl '^ofi' -n 20 python get_gapped_sf.py 50

echo "FINISHED"
