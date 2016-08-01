#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p cox
#SBATCH --gres=gpu
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --mem=110000
#SBATCH -t 10-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/home05/haehn/nets/out-rgbanet_border_cox.txt
#SBATCH -e /n/home05/haehn/nets/err-rgbanet_border_cox.txt

# add additional commands needed for Lmod and module loads here
source new-modules.sh
module load Anaconda/2.1.0-fasrc01
module load gcc/4.8.2-fasrc01 acml/5.3.1-fasrc01
module load gcc/4.8.2-fasrc01 openmpi/1.10.1-fasrc01 hdf5/1.8.16-fasrc01

module load cuda/7.5-fasrc01
module load cudnn/7.0-fasrc01

source /n/home05/haehn/nolearncox/bin/activate

# add commands for analyses here
cd /n/home05/haehn/Projects/mlproof/
python run_cnn.py -n RGBANet -p cylinder1_rgba -b border_overlap -d rectify

# end of program
exit 0;
