#!/bin/bash
# Set the model type (you'll loop through these)

MODEL_TYPE=$1

# Run the sweep script with the model type
python sweep.py --model_type "$MODEL_TYPE" --count 20  # Pass model_type and count