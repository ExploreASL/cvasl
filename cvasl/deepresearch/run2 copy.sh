#!/bin/bash

# slurm specific parameters
#SBATCH --job-name=aslbrainage
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=luna-gpu-long
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=6-23:59
#SBATCH --nice=0
#SBATCH --qos=radv
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=s.amiri@esciencecenter.nl
set -eu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load Anaconda3/2024.02-1
module load cuda/12.8
conda activate brainage

# Function to run training command and handle errors
run_training() {
  local model_type=$1
  local learning_rate=$2
  shift 2 # Shift past model_type and learning_rate

  local output_file="log"  # Log file
  local resnet_args=""
  local densenet_args=""
  local hybrid_cnn_args=""
  local resnext_args=""
  local efficientnet_args=""
  local large_cnn_args=""
  local improved_cnn_args=""
    local brainage_args=""

  echo "Training model: ${model_type} with learning rate: ${learning_rate}" >> "$output_file"

  # Conditionally set ResNet arguments

  if [[ "$model_type" == "efficientnet3d" ]]; then
    efficientnet_args="--efficientnet_dropout $1 --efficientnet_width_coefficient $2 --efficientnet_depth_coefficient $3 --efficientnet_initial_filters $4"  # Removed multiplier
    shift 4  # Shift past efficientnet specific args
  fi

  if [[ "$model_type" == "large" ]]; then
    large_cnn_args="--large_cnn_use_bn --large_cnn_use_se --large_cnn_use_dropout --large_cnn_dropout_rate $1 --large_cnn_layers $2 --large_cnn_filters $3 --large_cnn_filters_multiplier $4"
    shift 4 # Shift past the large cnn specific args
  fi

  if [[ "$model_type" == "improved_cnn" ]]; then
    improved_cnn_args="--improved_cnn_use_se --improved_cnn_dropout_rate $1 --improved_cnn_num_conv_layers $2 --improved_cnn_initial_filters $3 --improved_cnn_filters_multiplier $4"
    shift 4 # Shift past the improved cnn specific args
  fi

  if [[ "$model_type" == "hybrid_cnn_transformer" ]]; then
      hybrid_cnn_args="--hybrid_cnn_backbone_type $1 --hybrid_cnn_${1}_layers $2 $3 $4 --hybrid_cnn_initial_filters $5 --hybrid_transformer_layers $6 --hybrid_transformer_heads $7 --hybrid_transformer_ffn_dim $8 --hybrid_transformer_dropout $9"
    if [[ "$1" == "resnet" ]]; then
        hybrid_cnn_args+=" --hybrid_cnn_resnet_filters_multiplier ${10} --hybrid_cnn_resnet_use_se --hybrid_cnn_resnet_dropout"
    elif [[ "$1" == "densenet" ]]; then
         hybrid_cnn_args+=" --hybrid_cnn_densenet_growth_rate ${10} --hybrid_cnn_densenet_transition_filters_multiplier ${11} --hybrid_cnn_densenet_use_se --hybrid_cnn_densenet_dropout"
    fi
      shift 11
  fi

    # BrainAgeLoss arguments.
    local alpha=$1
    local beta=$2
    local gamma=$3
    local smoothing=$4
    local use_huber=$5
    echo "Use Huber: $use_huber", "Alpha: $alpha", "Beta: $beta", "Gamma: $gamma", "Smoothing: $smoothing"
    shift 5
    
    brainage_args="--brainage_alpha $alpha --brainage_beta $beta --brainage_gamma $gamma --brainage_smoothing $smoothing"

    if [[ "$use_huber" == "True" ]]; then
        brainage_args+=" --brainage_use_huber --brainage_delta $1"
        shift 1
    fi
    echo "Brainage args: $brainage_args"
    # No 'else' needed.  If use_huber is not "True", we just don't add the flag.


  # Construct the Python command.

  python train.py \
    --model_type "$model_type" \
    --csv_file "/home/radv/samiri/my-scratch/trainingdata/masked/topmri.csv" \
    --image_dir "/home/radv/samiri/my-scratch/trainingdata/masked/topmri" \
    --num_epochs 100 \
    --batch_size 10 \
    --learning_rate "$learning_rate" \
    --bins 20 \
    --use_wandb \
    --wandb_prefix lossopt \
    --use_cuda \
    --store_model \
    --split_strategy "stratified_group_sex" \
    --output_dir "/home/radv/samiri/my-scratch/saved_models" \
    $resnet_args $densenet_args $resnext_args $efficientnet_args $large_cnn_args $improved_cnn_args $hybrid_cnn_args $brainage_args "$@" 2>&1 | tee -a "$output_file"

  # Check exit status.
  if [ $? -ne 0 ]; then
    echo "Training FAILED for model: ${model_type} with learning rate: $learning_rate" >> "$output_file"
  else:
    echo "Training SUCCESSFUL for model: ${model_type} with learning rate: $learning_rate" >> "$output_file"
  fi
}

# Main script execution

# Create the output directory.
mkdir -p saved_models_test

# Clear the log file.
> log

# Learning rates to loop through
learning_rates=(0.0005 0.00045 0.00055)
learning_rates2=(0.0005 0.00045 0.00055)
# Common filter values (for consistent comparison)

# BrainAgeLoss parameter combinations

loss_params=(
  # Baseline (MAE) -  Keep this for comparison.
  "0.0 0.0 0.0 0.0 False"

  # Individual Component Tests (as before, but with Huber variants)

  # Exploring different Huber deltas (with a moderate combination)
  "0.5 0.2 0.1 0.0 True 0.5"  # Lower delta (more like MAE)
  "0.5 0.2 0.1 0.0 True 2.0"  # Higher delta (more robust)

  # Combined Effects (Varying Alpha - Correlation Loss)
  "0.1 0.1 0.1 0.0 False"  # Low alpha
  "2.0 0.1 0.1 0.0 False"  # Very high alpha (emphasize correlation)

  # Combined Effects (Varying Beta - Bias Regularization)
  "0.5 0.05 0.1 0.0 False" # Low Beta
  "0.5 1.0 0.1 0.0 False"  # Very High Beta (emphasize std dev matching)

  # Combined Effects (Varying Gamma - Age-Specific Weighting)
  "0.5 0.1 0.05 0.0 False" # low gamma
  "0.5 0.1 1.0 0.0 False"  # Very High Gamma (emphasize older ages)

  # Combined Effects (with Huber Loss, delta=1.0)
  "0.1 0.1 0.1 0.0 True 1.0"  # Low combination (Huber)
  "0.5 0.2 0.1 0.0 True 1.0"  # Moderate combination (Huber) - a good starting point
  "1.0 0.3 0.2 0.0 True 1.0"  # Higher combination (Huber) - Your "All" from before.
  "2.0 0.5 0.5 0.0 True 1.0" # High Alpha and Beta with Huber

  # Exploring Smoothing (with a moderate combination)
  "0.5 0.2 0.1 0.1 False"  # Low smoothing
  "0.5 0.2 0.1 0.5 False"    # high smoothing
  "0.5 0.2 0.1 0.1 True 1.0" # low smoothing huber
  "0.5 0.2 0.1 0.5 True 1.0"    # high smoothing huber
  
)
    # Loop through each learning rate
for lr in "${learning_rates[@]}"; do
    # --- Large CNN ---
    for dropout_rate in 0.05 0.1 0.2 0.3; do
        for layers in 4 5 6; do
            for filters in 8 20 64; do
                for multiplier in 1.5 2.5; do # Increased multipliers for large CNN
                    for loss_param in "${loss_params[@]}"; do
                      echo "Running large CNN with dropout rate: $dropout_rate, layers: $layers, filters: $filters, multiplier: $multiplier, loss params: $loss_param"
                      run_training "large" "$lr" $dropout_rate $layers $filters $multiplier $loss_param
                  done
                done
            done
        done
    done

    for dropout_rate in 0.05 0.1 0.2 0.3; do
        for layers in 6 7; do
            for filters in 8 40 ; do
                for multiplier in 0.8 1.1 1.5; do
                    for loss_param in "${loss_params[@]}"; do
                        echo "Running improved CNN with dropout rate: $dropout_rate, layers: $layers, filters: $filters, multiplier: $multiplier, loss params: $loss_param"
                        run_training "improved_cnn" "$lr" $dropout_rate $layers $filters $multiplier $loss_param
                    done
                done
            done
        done
    done
done
    


echo "All model training attempts completed.  See 'log' for details."