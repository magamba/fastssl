#!/usr/bin/env bash

#SBATCH -A berzelius-2023-242
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH --reservation 1g.10gb
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user mgamba@kth.se
#SBATCH --output /proj/memorization/logs/%A_%a.out
#SBATCH --error /proj/memorization/logs/%A_%a.err
#SBATCH --array=0-191%8

NAME="ssl_simclr"

# load env
source scripts/setup_env

export SLURM_TMPDIR="/scratch/local/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
if [ ! -d "$SLURM_TMPDIR" ]; then
    mkdir -p "$SLURM_TMPDIR"
fi

WANDB__SERVICE_WAIT=300

#dataset='stl10'
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
    jac_batch_size=4
    proj_str="simclr-stl10-"
    ckpt_str="-stl10"
else
    batch_size=512
    jac_batch_size=4
    proj_str="simclr-cifar10-"
    ckpt_str="-cifar10"
fi

label_noise=(
    0
#    5
#    10
#    15
#    20
#    40
#    60
#    80
#    100
)

noise=${label_noise[0]}
SEEDS=3
WIDTHS=64

width=$((1+SLURM_ARRAY_TASK_ID%WIDTHS))
seed=$((SLURM_ARRAY_TASK_ID/WIDTHS))

dataset="$1"
noise="$2"

#NOISE=${#label_noise[@]}
#SEEDS=3
#NCONFS=$((NOISE * SEEDS))

#width=$((1+SLURM_ARRAY_TASK_ID/NCONFS))
#conf=$((SLURM_ARRAY_TASK_ID%NCONFS))

#noise_id=$((conf % NOISE))
#noise=${label_noise[$noise_id]}
#seed=$((conf / NOISE))

echo "Model width: $width"
echo "Seed: $seed"
echo "Label noise: $noise"

num_workers=4

temperature=0.1
pdim=$(($width * 32))

wandb_group='smoothness'

## configure checkpointing dirs and dataset paths

wandb_projname="$proj_str"'ssl-effective_rank+overfit'
checkpt_dir="${SAVE_DIR}"/"$NAME""$ckpt_str"

src_checkpt="$checkpt_dir/resnet18/width$width/2_augs/temp_"$temperature"00_pdim_"$pdim"_no_autocast_bsz_"$batch_size"_lr_0.001_wd_1e-05/exp_SimCLR_100_seed_"$seed".pt"

if [ ! -f "$src_checkpt" ];
then
    echo "Error: no file not found $src_checkpt"
    exit 1
else
    echo "Copying SSL features to local storage"
    cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_ssl_100.pth"
fi

model=resnet18feat_width${width}

# evaluation for 0 label noise
if [ "$noise" = "0" ]; then

    # dataset locations
    trainset="${DATA_DIR}"/$dataset
    testset="${DATA_DIR}"/$dataset

    # running eval for 0 label noise
    # Let's precache features, should take ~35 seconds (rtx8000)
    python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                        --training.temperature=$temperature --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}_train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname
    status=$?

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                        --training.temperature=$temperature --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}_train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --training.log_interval=10 \
                        --training.track_jacobian=True \
                        --training.jacobian_batch_size=32 \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname
    new_status=$?
    status=$((status|new_status))

else

    # running eval with label noise
    wandb_projname="$proj_str"'ssl-effective_rank+overfit-noise'$noise
    checkpt_dir="${SAVE_DIR}"/"$NAME""_noise"$noise"$ckpt_str"

    if [ ! -d "$checkpt_dir" ]
    then
        mkdir -p "$checkpt_dir"
    fi

    # dataset locations
    trainset="${DATA_DIR}"/$dataset"-Noise_"$noise
    testset="${DATA_DIR}"/$dataset

    # Let's precache features, should take ~35 seconds (rtx8000)
    python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                        --training.temperature=$temperature --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}/train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --training.label_noise=$noise \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname
    status=$?

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                        --training.temperature=$temperature --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}/train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --training.log_interval=10 \
                        --training.label_noise=$noise \
                        --training.track_jacobian=True \
                        --training.jacobian_batch_size=32 \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname

    new_status=$?
    status=$((status|new_status))

fi

# save precached features to checkpt_dir/feats
if [ ! -d $checkpt_dir/feats ]
then
    mkdir $checkpt_dir/feats
fi

cp -r $SLURM_TMPDIR/feats/* $checkpt_dir/feats/

exit $status
