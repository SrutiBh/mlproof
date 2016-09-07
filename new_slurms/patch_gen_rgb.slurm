#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p cox
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=100000
#SBATCH -t 3-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/home05/haehn/nets/out-patchgen_rgb.txt
#SBATCH -e /n/home05/haehn/nets/err-patchgen_rgb.txt



# add commands for analyses here
cd /n/home05/haehn/Projects/mlproof/
python gen_patches_rgb.py

# end of program
exit 0;
