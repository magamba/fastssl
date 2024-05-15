#!/usr/bin/env bash

#SBATCH -A berzelius-2024-123
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH --reservation 1g.10gb
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user mgamba@kth.se
#SBATCH --output /proj/memorization/logs/%A_%a.out
#SBATCH --error /proj/memorization/logs/%A_%a.err
#SBATCH --array 0-2,3-5,9-11,21-23,45-47,69-71,93-95,141-143,189-191
####SBATCH --array 0-191%4

NAME="ssl_barlow_twins"

# load env
source scripts/setup_env

export SLURM_TMPDIR="/scratch/local/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
if [ ! -d "$SLURM_TMPDIR" ]; then
    mkdir -p "$SLURM_TMPDIR"
fi

WANDB__SERVICE_WAIT=300

dataset='cifar10c'
pretrain_dataset='cifar10'
batch_size=512
jac_batch_size=4
proj_str="bt-cifar10-"
ckpt_str="-cifar10"

noise_types=(
    "brightness"
    "defocus_blur"
    "frost"
    "glass_blur"
    "saturate"
    "spatter"
    "elastic_transform"
    "gaussian_blur"
    "impulse_noise"
    "motion_blur"
    "shot_noise"
    "speckle_noise"
    "contrast"
    "fog"
    "gaussian_noise"
    "jpeg_compression"
    "pixelate"
    "snow"
    "zoom_blur"
)

SEEDS=3
width=$((1+SLURM_ARRAY_TASK_ID/SEEDS))
seed=$((SLURM_ARRAY_TASK_ID%SEEDS))

echo "Model width: $width"
echo "Seed: $seed"

num_workers=16

lambd=0.005
#pdim=2048
pdim=$(($width * 32))

wandb_group='smoothness'

## configure checkpointing dirs and dataset paths

wandb_projname="$proj_str"'ssl-ood'
checkpt_dir="${SAVE_DIR}"/"$NAME""$ckpt_str"

src_checkpt="$checkpt_dir/resnet18/width"$width"/2_augs/lambd_"$lambd"000_pdim_"$pdim"_lr_0.001_wd_1e-05/exp_ssl_100_seed_"$seed".pt"

if [ ! -f "$src_checkpt" ];
then
    echo "Error: no file not found $src_checkpt"
    exit 1
else
    echo "Copying SSL features to local storage"
    cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_ssl_100.pth"
fi

model=resnet18feat_width${width}

# train linear probe on plain dataset

# dataset locations
trainset="${DATA_DIR}"/$pretrain_dataset
testset="${DATA_DIR}"/$pretrain_dataset

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
status=$?

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

src_checkpt="$checkpt_dir/resnet18/width"$width"/2_augs/lambd_"$lambd"000_pdim_"$pdim"_lr_0.001_wd_1e-06/1_augs_eval/exp_linear_200_seed_"$seed".pt"

if [ ! -f "$src_checkpt" ];
then
    echo "Error: file not found $src_checkpt"
    exit 1
else
    echo "Copying linear features to local storage"
    cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_ssl_200.pth"
fi

for noise in ${noise_types[@]}; do

    echo "Evaluating OOD noise type $noise"

    wandb_projname="$proj_str"'ssl-ood-'$noise
    checkpt_dir="${SAVE_DIR}"/"$NAME""$ckpt_str"

    if [ ! -d "$checkpt_dir" ]
    then
        mkdir -p "$checkpt_dir"
    fi

    # dataset locations
    trainset="${DATA_DIR}"/$pretrain_dataset
    testset="${DATA_DIR}"/cifar10-c/$noise

    # Let's precache features, should take ~35 seconds (rtx8000)
    python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                        --training.lambd=$lambd --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}_train.beton \
                        --training.val_dataset=${testset}/test.beton \
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
                        --training.val_dataset=${testset}/test.beton \
                        --training.log_interval=10 \
                        --training.track_jacobian=True \
                        --training.jacobian_batch_size=32 \
                        --eval.ood_eval=True \
                        --eval.ood_noise_type=$noise \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname

    new_status=$?
    status=$((status|new_status))

done

exit $status
