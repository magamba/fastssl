#!/usr/bin/env bash

#SBATCH -A berzelius-2023-229
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -C thin
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user mgamba@kth.se
#SBATCH --output /proj/memorization/logs/%A_%a.out
#SBATCH --error /proj/memorization/logs/%A_%a.err
#SBATCH --array=33-35
####SBATCH --array 39-41,69-71
####SBATCH --array=0-191%64

NAME="ssl_barlow_twins"

# load env
source scripts/setup_env

export SLURM_TMPDIR="/scratch/local/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
if [ ! -d "$SLURM_TMPDIR" ]; then
    mkdir -p "$SLURM_TMPDIR"
fi

WANDB__SERVICE_WAIT=300

dataset='stl10'
#dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
    jac_batch_size=8
    proj_str="bt-stl10x-"
    ckpt_str="-stl10"
else
    batch_size=512
    jac_batch_size=2
    proj_str="bt-cifar10-"
    ckpt_str="-cifar10"
fi

SEEDS=3

num_workers=16

width=$((1+SLURM_ARRAY_TASK_ID/SEEDS))
seed=$((SLURM_ARRAY_TASK_ID%SEEDS))
lambd=0.005
#pdim=2048
pdim=$(($width * 32))

wandb_group='smoothness'

model=resnet18proj_width${width}

## configure checkpointing dirs and dataset paths

wandb_projname="$proj_str"'ssl-effective_rank+overfit'
checkpt_dir="${SAVE_DIR}"/"$NAME""$ckpt_str"

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
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --training.num_workers=$num_workers \
                    --training.log_interval=10 \
                    --training.track_alpha=True \
                    --training.track_jacobian=True \
                    --training.jacobian_batch_size=$jac_batch_size \
                    --training.weight_decay=1e-5 \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname

status=$?

# let's save the model checkpoints to persistent storage
destdir=$checkpt_dir/resnet18/width${width}/2_augs/lambd_"$lambd"000_pdim_"$pdim"_lr_0.001_wd_1e-05/2_augs_train
if [ ! -d $destdir ]; then
    mkdir -p $destdir
fi
cp -v "$SLURM_TMPDIR/exp_ssl_100.pth" "$destdir/exp_ssl_100_seed_"$seed".pt"

new_status=$?
status=$((status|new_status))

echo "Done with training. Please run eval separately."
exit $status

model=resnet18feat_width${width}

# running eval for 0 label noise
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.num_workers=$num_workers \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname
new_status=$?
status=$((status|new_status))

# run linear eval on precached features from model: using default seed 42
python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
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

# save precached features to checkpt_dir/feats
if [ ! -d $checkpt_dir/feats ]
then
    mkdir $checkpt_dir/feats
fi

cp -r $SLURM_TMPDIR/feats/* $checkpt_dir/feats/

for noise in 5 10 15 20 40 60 80 100; do
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
                        --training.lambd=$lambd --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}/train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --training.label_noise=$noise \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname
    new_status=$?
    status=$((status|new_status))

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                        --training.lambd=$lambd --training.projector_dim=$pdim \
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

    # save precached features to checkpt_dir/feats
    if [ ! -d $checkpt_dir/feats ]
    then
        mkdir $checkpt_dir/feats
    fi

    cp -r $SLURM_TMPDIR/feats/* $checkpt_dir/feats/

done

exit $status
