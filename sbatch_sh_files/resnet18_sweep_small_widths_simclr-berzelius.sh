#! /bin/bash
#SBATCH -A berzelius-2024-341
#SBATCH --gpus=1
#SBATCH -t 3:00:00
#SBATCH --reservation safe
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user mgamba@kth.se
#SBATCH --output /proj/memorization/logs/%A_%a.out
#SBATCH --error /proj/memorization/logs/%A_%a.err
#SBATCH --array 51%8
#####SBATCH --array 45-59%8
####SBATCH --array 87-115
####SBATCH --array 0-173%30

NAME="ssl_simclr_robustness_no_compile"

# load env
source scripts/setup_env

if [ -z "$1" ]; then
    echo "Usage: $0 PROJECTOR_DEPTH [DATASET_SIZE_RATIO] [SEED] [NUM_PRETRAIN_AUGS]"
    exit 1
fi

export SLURM_TMPDIR="/scratch/local/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
if [ ! -d "$SLURM_TMPDIR" ]; then
    mkdir -p "$SLURM_TMPDIR"
fi

WANDB__SERVICE_WAIT=300

#dataset='stl10'
dataset='cifar10'
#dataset='cifar100'
if [ $dataset = 'stl10' ]
then
    batch_size=256
    jac_batch_size=8
    proj_str="simclr-stl10-"
    ckpt_str="-stl10"
elif [ $dataset = 'cifar100' ]; then
    batch_size=512
    jac_batch_size=512
    proj_str="simclr-cifar100-"
    ckpt_str="-cifar100"
else
    batch_size=512
    jac_batch_size=512
    proj_str="simclr-cifar10-"
    ckpt_str="-cifar10"
fi
pretrain_dataset="$dataset"

PRETRAIN=""
LINEAR_EVAL="Ture"
NOISY_EVAL=""
OOD_EVAL="True"
SSL_EVAL="True" # empty string to disable

temps=(0.005 0.02 0.05 0.1 0.2 0.5)
#pdepths=(1 2 3 4)
widths=({8..64..4})
pdepth=$1
dsize=$2
seed=$3
naugs=$4
epochs=100

if [ "$naugs" == "" ]; then
    naugs=2
fi

if [ "$dsize" == "" ] || [ "$dsize" == "0" ]; then
    dsize=0
else
    ckpt_str="$ckpt_str""-nsamples_""$dsize"
    dsize_int=$(python -c "print(round(float($dsize * 50000)))")
    if [ $dsize_int -lt $batch_size ]; then
        batch_size=$dsize_int
    fi
    if [ $dsize_int -lt $jac_batch_size ]; then
        jac_batch_size=$dsize_int
    fi
fi

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
num_workers=16
pdim=$(($width * 32))

if [ "$seed" == "" ]; then
    seed=0
fi

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
testset="${DATA_DIR}"/$dataset"_test.beton"
if [ "$dsize" != "0" ]; then
    trainset="${DATA_DIR}"/$dataset"-nsamples_$dsize"/train.beton
else
    trainset="${DATA_DIR}"/"$dataset"_train.beton
fi

if [ "$PRETRAIN" != "" ]; then
echo "Pretraining model"

# Let's train a SSL (SimCLR) model with the above hyperparams
python scripts/train_model_widthVary.py --config-file configs/cc_SimCLR.yaml \
                    --training.temperature=$temp --training.projector_dim=$pdim \
                    --training.projector_depth=$pdepth \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.train_dataset=${trainset} \
                    --training.val_dataset=${testset} \
                    --training.num_workers=$num_workers \
                    --training.log_interval=20 \
                    --training.track_alpha=True \
                    --training.track_jacobian=True \
                    --training.track_covariance=True \
                    --training.covariance_augmentations=10 \
                    --training.jacobian_batch_size=$jac_batch_size \
                    --training.weight_decay=1e-5 \
                    --training.algorithm="SimCLR" \
                    --training.epochs="$epochs" \
                    --training.num_augmentations=$naugs \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname

status=$?

# let's save the model checkpoints to persistent storage
destdir=$checkpt_dir/resnet18/width${width}/"$naugs"_augs/temp_"$(printf %.3f $temp)"_pdim_"$pdim"_pdepth_"$pdepth"_bsz_"$batch_size"_lr_0.001_wd_1e-05/"$naugs"_augs_train
if [ ! -d $destdir ]; then
    mkdir -p $destdir
fi
cp -v "$SLURM_TMPDIR/exp_SimCLR_"$epochs".pth" "$destdir/exp_SimCLR_"$epochs"_seed_"$seed".pt"

fi # end pretrain

src_checkpt="$checkpt_dir/resnet18/width"$width"/"$naugs"_augs/temp_"$(printf %.3f $temp)"_pdim_"$pdim"_pdepth_"$pdepth"_bsz_"$batch_size"_lr_0.001_wd_1e-05/"$naugs"_augs_train/exp_SimCLR_"$epochs"_seed_"$seed".pt"

if [ ! -f "$src_checkpt" ];
then
    echo "Error: file not found $src_checkpt"
    exit 1
else
    echo "Copying SSL features to local storage"
    cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_SimCLR_"$epochs".pth"
fi

new_status=$?
status=$((status|new_status))

model=resnet18feat_width${width}

if [ "$LINEAR_EVAL" != "" ]; then
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
                    --training.train_dataset=${trainset} \
                    --training.val_dataset=${testset} \
                    --eval.train_algorithm="SimCLR" \
                    --eval.num_augmentations_pretrain=$naugs \
                    --eval.epoch=$epochs \
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
                    --training.train_dataset=${trainset} \
                    --training.val_dataset=${testset} \
                    --training.log_interval=10 \
                    --training.track_jacobian=True \
                    --training.jacobian_batch_size=$jac_batch_size \
                    --eval.train_algorithm="SimCLR" \
                    --eval.num_augmentations_pretrain=$naugs \
                    --eval.epoch=$epochs \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname
new_status=$?
status=$((status|new_status))

fi # end linear eval

if [ "$NOISY_EVAL" != "" ]; then
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
    testset="${DATA_DIR}"/$dataset"_test.beton"
    if [ "$dsize" != "0" ]; then
        trainset="${DATA_DIR}"/$dataset"-nsamples_"$dsize"-Noise_"$noise"/train.beton"
    else
        trainset="${DATA_DIR}"/$dataset"-Noise_"$noise"/train.beton"
    fi

    # Let's precache features, should take ~35 seconds (rtx8000)
    python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                        --training.temperature=$temp --training.projector_dim=$pdim \
                        --training.projector_depth=$pdepth \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset} \
                        --training.val_dataset=${testset} \
                        --training.label_noise=$noise \
                        --eval.train_algorithm="SimCLR" \
                        --eval.epoch=$epochs \
                        --eval.num_augmentations_pretrain=$naugs \
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
                        --training.train_dataset=${trainset} \
                        --training.val_dataset=${testset} \
                        --training.log_interval=20 \
                        --training.label_noise=$noise \
                        --training.track_jacobian=True \
                        --training.jacobian_batch_size=$jac_batch_size \
                        --eval.train_algorithm="SimCLR" \
                        --eval.num_augmentations_pretrain=$naugs \
                        --eval.epoch=$epochs \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname

    new_status=$?
    status=$((status|new_status))

done

fi # end noisy eval

if [ "$OOD_EVAL" != "" ]; then
echo "OOD evaluation"

dataset="$pretrain_dataset""c"

checkpt_dir="${SAVE_DIR}"/"$NAME""$ckpt_str"

src_checkpt="$checkpt_dir/resnet18/width"$width"/"$naugs"_augs/temp_"$(printf %.3f $temp)"_pdim_"$pdim"_pdepth_"$pdepth"_bsz_"$batch_size"_lr_0.001_wd_1e-05/"$naugs"_augs_train/exp_SimCLR_"$epochs"_seed_"$seed".pt"

if [ ! -f "$src_checkpt" ];
then
    echo "Error: file not found $src_checkpt"
    exit 1
else
    echo "Copying SSL features to local storage"
    cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_ssl_"$epochs".pth"
fi


# dataset locations
testset="${DATA_DIR}"/$pretrain_dataset"_test.beton"
if [ "$dsize" != "" ] && [ "$dsize" != "0" ]; then
    trainset="${DATA_DIR}"/$pretrain_dataset"-nsamples_"$dsize"/train.beton"
else
    trainset="${DATA_DIR}"/$pretrain_dataset"_train.beton"
fi

# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                    --training.temperature=$temp --training.projector_dim=$pdim \
                    --training.projector_depth=$pdepth \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.num_workers=$num_workers \
                    --training.train_dataset=${trainset} \
                    --training.val_dataset=${testset} \
                    --eval.train_algorithm="SimCLR" \
                    --eval.num_augmentations_pretrain=$naugs \
                    --eval.epoch=$epochs \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname
new_status=$?
status=$((status|new_status))

src_checkpt="$checkpt_dir/resnet18/width"$width"/"$naugs"_augs/temp_"$(printf %.3f $temp)"_pdim_"$pdim"_pdepth_"$pdepth"_bsz_"$batch_size"_lr_0.001_wd_1e-06/1_augs_eval/exp_linear_200_seed_"$seed".pt"

if [ ! -f "$src_checkpt" ];
then
    echo "Error: file not found $src_checkpt"
    exit 1
else
    echo "Copying linear features to local storage"
    cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_SimCLR_200.pth"
fi


wandb_projname="$proj_str"'ssl-ood-'$noise
checkpt_dir="${SAVE_DIR}"/"$NAME""$ckpt_str"

if [ ! -d "$checkpt_dir" ]
then
    mkdir -p "$checkpt_dir"
fi

for noise in ${ood_noise_types[@]}; do

    # dataset locations
    testset="${DATA_DIR}"/"$pretrain_dataset"-c/$noise/test.beton
    if [ "$dsize" != "" ] && [ "$dsize" != "0" ]; then
        trainset="${DATA_DIR}"/$pretrain_dataset"-nsamples_"$dsize"/train.beton"
    else
        trainset="${DATA_DIR}"/$pretrain_dataset"_train.beton"
    fi

    # Let's precache features, should take ~35 seconds (rtx8000)
    python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                        --training.temperature=$temp --training.projector_dim=$pdim \
                        --training.projector_depth=$pdepth \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset} \
                        --training.val_dataset=${testset} \
                        --eval.train_algorithm="SimCLR" \
                        --eval.num_augmentations_pretrain=$naugs \
                        --eval.epoch=$epochs \
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
                        --training.train_dataset=${trainset} \
                        --training.val_dataset=${testset} \
                        --training.log_interval=10 \
                        --training.track_jacobian=True \
                        --training.jacobian_batch_size=$jac_batch_size \
                        --eval.train_algorithm="SimCLR" \
                        --eval.num_augmentations_pretrain=$naugs \
                        --eval.ood_eval=True \
                        --eval.ood_noise_type=$noise \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname

    new_status=$?
    status=$((status|new_status))
done

fi # end ood eval

if [ "$SSL_EVAL" != "" ]; then

    model=resnet18proj_width${width}

    # copy checkpoint of full model
    src_checkpt="$checkpt_dir/resnet18/width"$width"/"$naugs"_augs/temp_"$(printf %.3f $temp)"_pdim_"$pdim"_pdepth_"$pdepth"_bsz_"$batch_size"_lr_0.001_wd_1e-05/"$naugs"_augs_train/exp_SimCLR_"$epochs"_seed_"$seed".pt"

    if [ ! -f "$src_checkpt" ];
    then
        echo "Error: no file not found $src_checkpt"
        exit 1
    else
        echo "Copying SSL features to local storage"
        cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_ssl_"$epochs".pth"
    fi


    # dataset locations
    testset="${DATA_DIR}"/"$pretrain_dataset""_test.beton"
    if [ "$dsize" != "0" ]; then
        trainset="${DATA_DIR}"/"$pretrain_dataset""-nsamples_$dsize"/train.beton
    else
        trainset="${DATA_DIR}"/"$pretrain_dataset"_train.beton
    fi

    python scripts/train_model_widthVary.py --config-file configs/cc_SimCLR.yaml \
                        --training.temperature=$temp --training.projector_dim=$pdim \
                        --training.projector_depth=$pdepth \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.train_dataset=${trainset} \
                        --training.val_dataset=${testset} \
                        --training.num_workers=$num_workers \
                        --training.log_interval=20 \
                        --training.track_alpha=True \
                        --training.track_covariance=True \
                        --training.covariance_augmentations=10 \
                        --training.jacobian_batch_size=$jac_batch_size \
                        --training.weight_decay=1e-5 \
                        --training.num_augmentations=$naugs \
                        --training.algorithm="SimCLR" \
                        --eval.ssl_eval=True \
                        --eval.epoch=$epochs \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname

    new_status=$?
    status=$((status|new_status))

    # loop over noise
    # run ood eval without covariance
    for noise in ${ood_noise_types[@]}; do

        # dataset locations
        testset="${DATA_DIR}"/"$pretrain_dataset"-c/$noise/test.beton
        if [ "$dsize" != "" ] && [ "$dsize" != "0" ]; then
            trainset="${DATA_DIR}"/$pretrain_dataset"-nsamples_"$dsize"/train.beton"
        else
            trainset="${DATA_DIR}"/$pretrain_dataset"_train.beton"
        fi

        python scripts/train_model_widthVary.py --config-file configs/cc_SimCLR.yaml \
                            --training.temperature=$temp --training.projector_dim=$pdim \
                            --training.projector_depth=$pdepth \
                            --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                            --training.batch_size=$batch_size --training.model=$model \
                            --training.seed=$seed \
                            --training.train_dataset=${trainset} \
                            --training.val_dataset=${testset} \
                            --training.num_workers=$num_workers \
                            --training.log_interval=20 \
                            --training.track_alpha=True \
                            --training.jacobian_batch_size=$jac_batch_size \
                            --training.weight_decay=1e-5 \
                            --training.num_augmentations=$naugs \
                            --training.algorithm="SimCLR" \
                            --eval.ssl_eval=True \
                            --eval.ood_noise_type=$noise \
                            --eval.epoch=$epochs \
                            --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                            --logging.wandb_project=$wandb_projname

        new_status=$?
        status=$((status|new_status))
    done
fi # end SSL eval

exit $status
