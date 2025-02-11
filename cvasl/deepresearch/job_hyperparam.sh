#!/bin/bash

# List of all model types to sweep
MODEL_TYPES=("large" "improved_cnn" "resnet" "densenet" "resnext3d" "efficientnet3d" "hybrid_cnn_transformer")

for model_type in "${MODEL_TYPES[@]}"; do
  sh run_sweep.sh "$model_type"
done