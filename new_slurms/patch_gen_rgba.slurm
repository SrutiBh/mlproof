#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p cox
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=120000
#SBATCH -t 7-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/home05/haehn/nets/out-patchgen_rgba.txt
#SBATCH -e /n/home05/haehn/nets/err-patchgen_rgba.txt

source new-modules.sh
module load Anaconda/2.5.0-fasrc01
module load gcc/4.9.0-fasrc01

module load cuda/7.5-fasrc01
module load cudnn/7.0-fasrc01

module load opencv/3.0.0-fasrc04

# custom HDF5 lib
export LIBRARY_PATH=/n/home05/haehn/nolearncox/src/hdf5-1.8.17/hdf5/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/n/home05/haehn/nolearncox/src/hdf5-1.8.17/hdf5/lib:$LD_LIBRARY_PATH
export CPATH=/n/home05/haehn/nolearncox/src/hdf5-1.8.17/hdf5/include:$CPATH
export FPATH=/n/home05/haehn/nolearncox/src/hdf5-1.8.17/hdf5/include:$FPATH

source /n/home05/haehn/nolearncox/bin/activate

# add commands for analyses here
cd /n/home05/haehn/Projects/mlproof/
python gen_patches_rgba.py

# end of program
exit 0;
