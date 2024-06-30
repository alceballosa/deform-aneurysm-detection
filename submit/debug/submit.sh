set -x
set -e

# global variables
CPU_PER_GPU=16
# memories per gpu
MEM_PER_GPU=200
# the nice value for scheduling jobs
NICE=0

QUEUE=$1

# changable parameters
GPU_TYPE=$2
NUM_GPU=$3

NUM_CPU=$4
NUM_MEM=$5


# the name of the job
BASH_SCRIPT=$6
OPTS=${@:7}

BASH_SCRIPT_NAME=`basename ${BASH_SCRIPT} .sh`

JOB_NAME=${BASH_SCRIPT_NAME}
LOG_PREFIX=/work/vig/hieu

# do some simple calculation
# TASK_PER_NODE=`python3 -c "from math import ceil; print(ceil($GPU_PER_NODE / $NUM_GPU))"`
TASK_PER_NODE=1
# NUM_CPU=$((${NUM_GPU} * ${CPU_PER_GPU}))
# NUM_MEM=$((${NUM_GPU} * ${MEM_PER_GPU}))

sbatch --partition=${QUEUE} \
       --cpus-per-task=${NUM_CPU} \
       --mem=${NUM_MEM}G \
       --ntasks-per-node=${TASK_PER_NODE} \
       --gres=gpu:${GPU_TYPE}:${NUM_GPU} \
       --output="${LOG_PREFIX}_j%A.out" \
       --error="${LOG_PREFIX}_j%A.err" \
       --job-name=${JOB_NAME} \
       --signal=B:USR1@600 \
       --nice=${NICE} \
       --exclusive=user \
       --mail-user=zhu.fang@northeastern.edu \
       --mail-type=FAIL,END \
       --time=120:00:00 \
       ${BASH_SCRIPT} $OPTS