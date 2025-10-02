#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=multigpu
#SBATCH --time=24:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./logs/exec.%j.%x.out
#SBATCH --error=./logs/exec.%j.%x.out
#SBATCH --nice=0

export NUM_GPUS=2
export CONFIG_NAME=${SLURM_JOB_NAME}

echo $SLURM_JOB_NAME;
echo $1;
echo $2;

#module load cuda/11.3
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate cta2

WORKSPACE_PATH="/projects/vig/alberto/medical/deform-aneurysm-detection"

export PYTHONPATH=${WORKSPACE_PATH}
export PYTHONPATH=$(pwd):$PYTHONPATH



mkdir /dev/shm/internal_train 

/usr/sbin/sshd -D -p 2219 -f /dev/null -h ${HOME}/.ssh/alberto_neu &

echo "Copying data in background...";
#{ cp -r  /scratch/ceballosarroyo.a/aneurysm/cta_datasets/internal_train/crop_0.4  /dev/shm/internal_train/crop_0.4; } & { cp -r  /scratch/ceballosarroyo.a/aneurysm/cta_datasets/internal_train/vein_mask_edt_comp /dev/shm/internal_train/vein_mask_edt_comp; } & { cp -r  /scratch/ceballosarroyo.a/aneurysm/cta_datasets/internal_train/crop_0.4_vessel_edt_comp /dev/shm/internal_train/crop_0.4_vessel_edt_comp; } &

if [ ! -d /dev/shm/internal_train/crop_0.4 ]; then
    echo "Copying data to /dev/shm/internal_train";
    cp -r  /scratch/ceballosarroyo.a/aneurysm/cta_datasets/internal_train/crop_0.4  /dev/shm/internal_train/crop_0.4
    cp -r  /scratch/ceballosarroyo.a/aneurysm/cta_datasets/internal_train/crop_0.4_vessel_edt_comp /dev/shm/internal_train/crop_0.4_vessel_edt_comp
    #cp -r  /scratch/ceballosarroyo.a/aneurysm/cta_datasets/internal_train/vein_mask_edt_comp /dev/shm/internal_train/vein_mask_edt_comp
else
    echo "Data already copied to /dev/shm/internal_train";
fi


# sleep 7m;

export ID_PORT=$(($RANDOM+20010))
python src/train_net.py\
    --num-gpus $NUM_GPUS\
    --config-file "./configs/$1/$2.yaml"\
        --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --resume\


