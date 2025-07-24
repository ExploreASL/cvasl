#!/bin/bash

# slurm specific parameters
#SBATCH --job-name=aslbrainage
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=luna-gpu-long
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=6-23:59
#SBATCH --nice=0
#SBATCH --qos=radv
#SBATCH --mail-type=BEGIN

set -eu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CSV_FILE="/home/user/training.csv"
IMAGE_DIR="/home/user/training"
NUM_EPOCHS=100
SPLIT_STRATEGY="stratified_group_sex"
BATCH_SIZE=10
LEARNING_RATE=0.0005
BINS=20
OUTPUT_DIR="/home/user/saved_models"

while [[ $# -gt 0 ]]; do
  case $1 in
    --csv_file)
      CSV_FILE="$2"
      shift 2
      ;;
    --image_dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --split_strategy)
      SPLIT_STRATEGY="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --bins)
      BINS="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --csv_file CSV_FILE           Path to CSV file (default: $CSV_FILE)"
      echo "  --image_dir IMAGE_DIR         Path to image directory (default: $IMAGE_DIR)"
      echo "  --num_epochs NUM_EPOCHS       Number of training epochs (default: $NUM_EPOCHS)"
      echo "  --split_strategy STRATEGY     Data split strategy (default: $SPLIT_STRATEGY)"
      echo "  --batch_size BATCH_SIZE       Training batch size (default: $BATCH_SIZE)"
      echo "  --learning_rate LEARNING_RATE Learning rate (default: $LEARNING_RATE)"
      echo "  --bins BINS                   Number of bins (default: $BINS)"
      echo "  --output_dir OUTPUT_DIR       Output directory for models (default: $OUTPUT_DIR)"
      echo "  -h, --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

module load Anaconda3/2024.02-1
module load cuda/12.8
conda activate brainage

run_training() {
  local model_type=$1
  local learning_rate=$2
  shift 2 

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
    shift 5
    brainage_args="--brainage_alpha $alpha --brainage_beta $beta --brainage_gamma $gamma --brainage_smoothing $smoothing"

    if [[ "$use_huber" == "True" ]]; then
        brainage_args+=" --brainage_use_huber --brainage_delta $1"
        shift 1
    fi
    


  
  python train.py \
    --model_type "$model_type" \
    --csv_file "$CSV_FILE" \
    --image_dir "$IMAGE_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$learning_rate" \
    --bins "$BINS" \
    --use_wandb \
    --wandb_prefix lossopt \
    --use_cuda \
    --store_model \
    --split_strategy "$SPLIT_STRATEGY" \
    --output_dir "$OUTPUT_DIR" \
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
learning_rates=(0.0005)
learning_rates2=(0.0005)


loss_params=(
  # Baseline (MAE)
  "0.0 0.0 0.0 0.0 False"

  # Individual Component Tests (as before, but with Huber variants)

  # Exploring different Huber deltas (with a moderate combination)
  "0.5 0.2 0.1 0.0 True 0.5"  # Lower delta (more like MAE)
  "0.5 0.2 0.1 0.0 True 2.0"  # Higher delta (more robust)

  # Combined Effects (Varying Alpha - Corerlation Loss)
  "0.1 0.1 0.1 0.0 False"  # low Alpha
  "2.0 0.1 0.1 0.0 False"  # Very high alpha (emphasize correlation)

  # Combined Effects (Varying Beta - Bias Regularization)
  "0.5 0.05 0.1 0.0 False" # Low BEta
  "0.5 1.0 0.1 0.0 False"  # Very High Beta (emphasize std dev matching)

  # Combined Effects (Varying Gamma - Age-Specific Weighting)
  "0.5 0.1 0.05 0.0 False" # low gamma
  "0.5 0.1 1.0 0.0 False"  # Very High Gamma (emphasize older ages)

  # Combined Effects (with Huber Loss, delta=1.0)
  "0.1 0.1 0.1 0.0 True 1.0"  # Low Combination (Huber)
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
    for dropout_rate in 0.05 0.08; do
        for layers in 4 6; do
            for filters in 8; do
                for multiplier in 1.5; do
                    for loss_param in "${loss_params[@]}"; do
                      echo "Running large CNN with dropout rate: $dropout_rate, layers: $layers, filters: $filters, multiplier: $multiplier, loss params: $loss_param"
                      run_training "large" "$lr" $dropout_rate $layers $filters $multiplier $loss_param
                  done
                done
            done
        done
    done

    for dropout_rate in 0.05 0.1; do
        for layers in 6 8; do
            for filters in 8 10; do
                for multiplier in 0.8; do
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
