#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p serial_requeue
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=160000
#SBATCH -t 1-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/home05/haehn/SLURM/patchgen/out-IPLB.txt
#SBATCH -e /n/home05/haehn/SLURM/patchgen/err-IPLB.txt

# add additional commands needed for Lmod and module loads here
source new-modules.sh

module load Anaconda/2.5.0-fasrc01
module load gcc/4.9.0-fasrc01
export CUDA_HOME=/usr/local/cuda-7.0
export CUDA_LIB=/usr/local/cuda-7.0/lib64
export CUDA_INCLUDE=/usr/local/cuda-7.0/include
export PATH=/usr/local/cuda-7.0/bin:$PATH
export CPATH=/usr/local/cuda-7.0/include:$CPATH
export FPATH=/usr/local/cuda-7.0/include:$FPATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-7.0/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LIBRARY_PATH

module load cudnn/7.0-fasrc01
module load jpeg/6b-fasrc01
module load opencv/3.0.0-fasrc04

source activate AETROS


# add commands for analyses here
cd /n/home05/haehn/Projects/mlproof/
python pg_rgba_iplb.py

# end of program
exit 0;
