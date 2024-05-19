#! /bin/bash
#SBATCH -A berzelius-2024-116
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH --reservation safe
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user mgamba@kth.se
#SBATCH --output /proj/memorization/logs/%A_%a.out
#SBATCH --error /proj/memorization/logs/%A_%a.err
#SBATCH --array 0-173%30

NAME="ssl_simclr_robustness"

# load env
source scripts/setup_env

if [ -z "$1" ]; then
    echo "Usage: $0 PROJECTOR_DEPTH"
    exit 1
fi

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
    jac_batch_size=8
    proj_str="simclr-stl10-"
    ckpt_str="-stl10"
else
    batch_size=512
    jac_batch_size=512
    proj_str="simclr-cifar10-"
    ckpt_str="-cifar10"
fi

temps=(0.005 0.02 0.05 0.1 0.2 0.5)
#pdepths=(1 2 3 4)
widths=({8..64..2})
pdepth=$1

ood_noise_types=(
    "frost"
    "glass_blur"
    "spatter"
    "gaussian_blur"
    "impulse_noise"
    "motion_blur"
    "shot_noise"
    "speckle_noise"
    "fog"
    "gaussian_noise"
    "jpeg_compression"
    "pixelate"
    "snow"
)

WIDTHS=${#widths[@]}
conf_id=$((SLURM_ARRAY_TASK_ID/WIDTHS))
width_id=$((SLURM_ARRAY_TASK_ID%WIDTHS))

width=${widths[width_id]}
temp=${temps[conf_id]}
seed=0
num_workers=16
pdim=$(($width * 32))

wandb_group='smoothness'

model=resnet18proj_width${width}

## configure checkpointing dirs and dataset paths

wandb_projname="$proj_str"'ssl-robustness'
checkpt_dir="${SAVE_DIR}"/"$NAME""$ckpt_str"

if [ ! -d "$checkpt_dir" ]
then
    mkdir -p "$checkpt_dir"
fi

# dataset locations
trainset="${DATA_DIR}"/$dataset
testset="${DATA_DIR}"/$dataset

echo "Pretraining model"

# Let's train a SSL (SimCLR) model with the above hyperparams
python scripts/train_model_widthVary.py --config-file configs/cc_SimCLR.yaml \
                    --training.temperature=$temp --training.projector_dim=$pdim \
                    --training.projector_depth=$pdepth \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --training.num_workers=$num_workers \
                    --training.log_interval=20 \
                    --training.track_alpha=True \
                    --training.track_jacobian=True \
                    --training.track_covariance=True \
                    --training.jacobian_batch_size=$jac_batch_size \
                    --training.weight_decay=1e-5 \
                    --training.algorithm="SimCLR" \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname

status=$?

# let's save the model checkpoints to persistent storage
destdir=$checkpt_dir/resnet18/width${width}/2_augs/temp_"$(printf %.3f $temp)"_pdim_"$pdim"_pdepth_"$pdepth"_bsz_"$batch_size"_lr_0.001_wd_1e-05/2_augs_train
if [ ! -d $destdir ]; then
    mkdir -p $destdir
fi
cp -v "$SLURM_TMPDIR/exp_SimCLR_100.pth" "$destdir/exp_SimCLR_100_seed_"$seed".pt"

src_checkpt="$checkpt_dir/resnet18/width"$width"/2_augs/temp_"$(printf %.3f $temp)"_pdim_"$pdim"_pdepth_"$pdepth"_bsz_"$batch_size"_lr_0.001_wd_1e-05/2_augs_train/exp_SimCLR_100_seed_"$seed".pt"

if [ ! -f "$src_checkpt" ];
then
    echo "Error: file not found $src_checkpt"
    exit 1
else
    echo "Copying SSL features to local storage"
    cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_SimCLR_100.pth"
fi

new_status=$?
status=$((status|new_status))

model=resnet18feat_width${width}

echo "Precaching features"

# running eval for 0 label noise
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                    --training.temperature=$temp --training.projector_dim=$pdim \
                    --training.projector_depth=$pdepth \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.num_workers=$num_workers \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --eval.train_algorithm="SimCLR" \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname
new_status=$?
status=$((status|new_status))

echo "Linear probe training"

# run linear eval on precached features from model: using default seed 42
python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                    --training.temperature=$temp --training.projector_dim=$pdim \
                    --training.projector_depth=$pdepth \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.num_workers=$num_workers \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --training.log_interval=10 \
                    --training.track_jacobian=True \
                    --training.jacobian_batch_size=512 \
                    --eval.train_algorithm="SimCLR" \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname
new_status=$?
status=$((status|new_status))


echo "Noisy labels training"

for noise in 10 20 40 60 80 100; do
    # running eval with label noise
    wandb_projname="$proj_str"'ssl-robustness-noise'$noise
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
                        --training.temperature=$temp --training.projector_dim=$pdim \
                        --training.projector_depth=$pdepth \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}/train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --training.label_noise=$noise \
                        --eval.train_algorithm="SimCLR" \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname
    new_status=$?
    status=$((status|new_status))

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                        --training.temperature=$temp --training.projector_dim=$pdim \
                        --training.projector_depth=$pdepth \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}/train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --training.log_interval=20 \
                        --training.label_noise=$noise \
                        --training.track_jacobian=True \
                        --training.jacobian_batch_size=512 \
                        --eval.train_algorithm="SimCLR" \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname

    new_status=$?
    status=$((status|new_status))

done

echo "OOD evaluation"

dataset='cifar10c'
pretrain_dataset='cifar10'

checkpt_dir="${SAVE_DIR}"/"$NAME""$ckpt_str"

src_checkpt="$checkpt_dir/resnet18/width"$width"/2_augs/temp_"$(printf %.3f $temp)"_pdim_"$pdim"_pdepth_"$pdepth"_bsz_"$batch_size"_lr_0.001_wd_1e-05/2_augs_train/exp_SimCLR_100_seed_"$seed".pt"

if [ ! -f "$src_checkpt" ];
then
    echo "Error: file not found $src_checkpt"
    exit 1
else
    echo "Copying SSL features to local storage"
    cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_ssl_100.pth"
fi


# dataset locations
trainset="${DATA_DIR}"/$pretrain_dataset
testset="${DATA_DIR}"/$pretrain_dataset

# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                    --training.temperature=$temp --training.projector_dim=$pdim \
                    --training.projector_depth=$pdepth \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.num_workers=$num_workers \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --eval.train_algorithm="SimCLR" \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname
new_status=$?
status=$((status|new_status))

src_checkpt="$checkpt_dir/resnet18/width"$width"/2_augs/temp_"$(printf %.3f $temp)"_pdim_"$pdim"_pdepth_"$pdepth"_bsz_"$batch_size"_lr_0.001_wd_1e-06/1_augs_eval/exp_linear_200_seed_"$seed".pt"

if [ ! -f "$src_checkpt" ];
then
    echo "Error: file not found $src_checkpt"
    exit 1
else
    echo "Copying linear features to local storage"
    cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_ssl_200.pth"
fi


wandb_projname="$proj_str"'ssl-ood-'$noise
checkpt_dir="${SAVE_DIR}"/"$NAME""$ckpt_str"

if [ ! -d "$checkpt_dir" ]
then
    mkdir -p "$checkpt_dir"
fi

#for noise in ${ood_noise_types[@]}; do
for noise in frost; do

    # dataset locations
    trainset="${DATA_DIR}"/$pretrain_dataset
    testset="${DATA_DIR}"/cifar10-c/$noise

    # Let's precache features, should take ~35 seconds (rtx8000)
    python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                        --training.temperature=$temp --training.projector_dim=$pdim \
                        --training.projector_depth=$pdepth \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}_train.beton \
                        --training.val_dataset=${testset}/test.beton \
                        --eval.train_algorithm="SimCLR" \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname

    new_status=$?
    status=$((status|new_status))

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                        --training.temperature=$temp --training.projector_dim=$pdim \
                        --training.projector_depth=$pdepth \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}_train.beton \
                        --training.val_dataset=${testset}/test.beton \
                        --training.log_interval=10 \
                        --training.track_jacobian=True \
                        --training.jacobian_batch_size=512 \
                        --eval.train_algorithm="SimCLR" \
                        --eval.ood_eval=True \
                        --eval.ood_noise_type=$noise \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname

    new_status=$?
    status=$((status|new_status))
done

exit $status
