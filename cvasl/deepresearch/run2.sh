#!/bin/bash

# Function to run training command and handle errors
run_training() {
  local model_type=$1
  shift  # Shift to remove the model_type from the positional arguments
  local output_file="log"  # Log file
  local resnet_args=""
  local densenet_args=""
  local hybrid_cnn_args=""
  local resnext_args="" # initialize
  local efficientnet_args="" # initialize
  local large_cnn_args=""
  local improved_cnn_args=""


  echo "Training model: ${model_type}" >> "$output_file"

  # Conditionally set ResNet arguments
  if [[ "$model_type" == "resnet" ]]; then
    resnet_args="--resnet_layers $1 $2 $3 --resnet_initial_filters $4 --resnet_use_se --resnet_dropout --resnet_filters_multiplier $5"
    shift 5 # Shift past the resnet specific args
  fi

  if [[ "$model_type" == "densenet" ]]; then
    densenet_args="--densenet_use_se --densenet_dropout --densenet_layers $1 $2 --densenet_growth_rate $3 --densenet_initial_filters $4 --densenet_transition_filters_multiplier $5"
     shift 5 # Shift past the densenet specific args

  fi
    if [[ "$model_type" == "resnext3d" ]]; then
        resnext_args="--resnext_use_se --resnext_dropout  --resnext_layers $1 $2 $3 --resnext_initial_filters $4 --resnext_cardinality $5 --resnext_bottleneck_width $6 --resnext_filters_multiplier $7"
        shift 7 # Shift past resnext specific args
    fi

    if [[ "$model_type" == "efficientnet3d" ]]; then
      efficientnet_args="--efficientnet_dropout $1 --efficientnet_width_coefficient $2 --efficientnet_depth_coefficient $3 --efficientnet_initial_filters $4 --efficientnet_filters_multiplier $5"
      shift 5  # Shift past efficientnet specific args

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
    shift 11 # important! shift by the number of arguments.

  fi

  # Construct the Python command.
  python train.py \
    --model_type "$model_type" \
    --csv_file "./trainingdata/mock_dataset/mock_data.csv" \
    --image_dir "./trainingdata/mock_dataset/images" \
    --num_epochs 2 \
    --batch_size 10 \
    --learning_rate 0.00001 \
    --bins 2 \
    --split_strategy "stratified_group_sex" \
    --output_dir "./saved_models_test" \
    $resnet_args $densenet_args $resnext_args $efficientnet_args $large_cnn_args $improved_cnn_args $hybrid_cnn_args "$@" 2>&1 | tee -a "$output_file" # Include resnet_args

  # Check the exit status.
  if [ $? -ne 0 ]; then
    echo "Training FAILED for model: ${model_type}" >> "$output_file"
  else:
    echo "Training SUCCESSFUL for model: ${model_type}" >> "$output_file"
  fi
}

# Main script execution

# Create the output directory.
mkdir -p saved_models_test

# Clear the log file.
> log

# # --- Test Basic Models ---
# # Large CNN
for i in $(seq 1 2); do
    dropout_rate=$(echo "scale=2; $i/20" | bc)  # 0.05 to 0.5
    layers=$((1 + i % 4))        # 1 to 4
    filters=$((8 * i))             # 8 to 80
    multiplier=$(echo "scale=2; 0.8 + $i/5" | bc ) # 0.8, 1.0, 1.2,... 2.6
    run_training "large" $dropout_rate $layers $filters $multiplier
done

# Improved CNN
for i in $(seq 1 2); do
    dropout_rate=$(echo "scale=2; $i/20" | bc)  # 0.05 to 0.5
    layers=$((1 + i % 4))          # 1 to 4
    filters=$((8 * i))      #8 to 80
    multiplier=$(echo "scale=2; 0.8 + $i/5" | bc )
    run_training "improved_cnn" $dropout_rate $layers $filters $multiplier
done

# # ResNet
for i in $(seq 1 2); do
    layers1=$((1 + i % 3))      # 1 to 3
    layers2=$((1 + (i+1) % 3))  # 1 to 3
    layers3=$((1 + (i+2) % 3))  # 1 to 3
    filters=$((8 * i))      #8 to 80
    multiplier=$(echo "scale=2; 0.8 + $i/5" | bc ) # 0.8, 1.0, 1.2, 1.4
    run_training "resnet" $layers1 $layers2 $layers3 $filters $multiplier
done

# # DenseNet
for i in $(seq 1 2); do
    layers1=$((1 + i % 4))          # 1 to 4
    layers2=$((1 + (i+1) % 4))      # 1 to 4
    growth_rate=$((4 * i)) #4, 8, 12, 16... 40
    filters=$((8 * i))        #8 to 80
    multiplier=$(echo "scale=2; 0.8 + $i/5" | bc )
    run_training "densenet" $layers1 $layers2 $growth_rate $filters $multiplier
done


# # ResNeXt3D
for i in $(seq 1 2); do
    layers1=$((1 + i % 3)) #1, 2, 3
    layers2=$((1 + (i + 1) % 3)) #1, 2, 3
    layers3=$((1 + (i+2) % 3))   #1,2,3
    filters=$((8 * i))
    cardinality=$((4 + 4*(i%8))) # 4, 8, 12, ..., 32
    bottleneck=$((1 + i%4)) #1, 2, 3, 4
    multiplier=$(echo "scale=2; 0.8 + $i/5" | bc )
    run_training "resnext3d" $layers1 $layers2 $layers3 $filters $cardinality $bottleneck $multiplier
done

# #EfficientNet3D

for i in $(seq 1 2); do
    dropout=$(echo "scale=2; $i/20" | bc)  # from 0.05 to 0.5
    width=$(echo "scale=2; 0.6 + $i/10" | bc)  # 0.6, 0.7, 0.8, ... 1.5
    depth=$(echo "scale=2; 0.6 + $i/10" | bc)  # 0.6, 0.7 ... 1.5
    initial_filters=$(( 8 * i))  #8 to 80
    multiplier=$(echo "scale=2; 0.8 + $i/5" | bc )

    run_training "efficientnet3d" $dropout $width $depth $initial_filters $multiplier
done

echo "All model training attempts completed.  See 'log' for details."