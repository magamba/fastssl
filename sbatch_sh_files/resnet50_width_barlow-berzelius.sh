#!/usr/bin/env bash

#SBATCH -A berzelius-2023-44
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -C fat
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user mgamba@kth.se
#SBATCH --output /proj/memorization/logs/%A_%a.out
#SBATCH --error /proj/memorization/logs/%A_%a.err
#SBATCH --array=0-575%64

NAME="ssl_barlow_twins"

# load env
source scripts/setup_env

export SLURM_TMPDIR="/scratch/local"

WANDB__SERVICE_WAIT=300

dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

SEEDS=3

num_workers=16

width=$((1+SLURM_ARRAY_TASK_ID/SEEDS))
width=$((width + 64))
seed=$((SLURM_ARRAY_TASK_ID%SEEDS))
#lambd=0.00397897
lambd=0.001
pdim=3072
wandb_group='smoothness'

model=resnet50proj_width${width}

## configure checkpointing dirs and dataset paths

wandb_projname='ssl-large_widths-overfit'
checkpt_dir="${SAVE_DIR}"/"$NAME"

if [ ! -d "$checkpt_dir" ]
then
    mkdir -p "$checkpt_dir"
fi

# dataset locations
trainset="${DATA_DIR}"/$dataset
testset="${DATA_DIR}"/$dataset

# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model_widthVary.py --config-file configs/cc_barlow_twins.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.weight_decay=1e-6 \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --training.num_workers=$num_workers \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname

status="$?"

model=resnet50feat_width${width}

# running eval for 0 label noise
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.num_workers=$num_workers \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton

# run linear eval on precached features from model: using default seed 42
python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.num_workers=$num_workers \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname

status="$?"

# save precached features to checkpt_dir/feats
if [ ! -d $checkpt_dir/feats ]
then
    mkdir $checkpt_dir/feats
fi

cp -r $SLURM_TMPDIR/feats/* $checkpt_dir/feats/

# running eval for 15% label noise
wandb_projname='ssl-large_widths-noise15-overfit'
checkpt_dir="${SAVE_DIR}"/"$NAME""_noise15"

if [ ! -d "$checkpt_dir" ]
then
    mkdir -p "$checkpt_dir"
fi

# dataset locations
trainset="${DATA_DIR}"/$dataset"-Noise_15"/$dataset
testset="${DATA_DIR}"/$dataset

# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.num_workers=$num_workers \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton

# run linear eval on precached features from model: using default seed 42
python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.num_workers=$num_workers \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname

status="$?"

# save precached features to checkpt_dir/feats
if [ ! -d $checkpt_dir/feats ]
then
    mkdir $checkpt_dir/feats
fi

cp -r $SLURM_TMPDIR/feats/* $checkpt_dir/feats/

exit $status
