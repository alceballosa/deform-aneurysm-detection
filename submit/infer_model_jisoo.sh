#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=jiang
#SBATCH --time=24:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./logs/exec.%j.%x.out
#SBATCH --error=./logs/exec.%j.%x.out
#SBATCH --nice=0

#module load cuda/11.3
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate cta2

WORKSPACE_PATH="/projects/vig/alberto/medical/deform-aneurysm-detection"

cd $WORKSPACE_PATH

export NUM_GPU=2

./run_several_trx_jisoo.sh $1 $2 $NUM_GPU

