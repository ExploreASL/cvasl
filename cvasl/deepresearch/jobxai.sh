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

DEFAULT_TRAINING_CSV="./data/training.csv"
DEFAULT_TEST_CSVS="./data/test1.csv ./data/test2.csv"
DEFAULT_TRAINING_IMG_DIR="./data/training_images"
DEFAULT_TEST_IMG_DIRS="./data/test1_images ./data/test2_images"
DEFAULT_MODEL_DIR="./saved_models"
DEFAULT_MASK_PATHS="./data/masks"
DEFAULT_STATS_OUTPUT_ROOT="./statistical_results"
DEFAULT_INDICES_PATH="./data/test_indices.npy"
DEFAULT_XAI_OUTPUT_DIR="./xai_results"
DEFAULT_XAI_METHOD="all"
DEFAULT_DEVICE="cuda"
DEFAULT_ATLAS_PATH="./atlases/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"
DEFAULT_INDIVIDUAL_PATIENTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --training-csv)
            TRAINING_CSV="$2"
            shift 2
            ;;
        --test-csvs)
            TEST_CSVS="$2"
            shift 2
            ;;
        --training-img-dir)
            TRAINING_IMG_DIR="$2"
            shift 2
            ;;
        --test-img-dirs)
            TEST_IMG_DIRS="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --mask-paths)
            MASK_PATHS="$2"
            shift 2
            ;;
        --stats-output-root)
            STATS_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --indices-path)
            INDICES_PATH="$2"
            shift 2
            ;;
        --xai-output-dir)
            XAI_OUTPUT_DIR="$2"
            shift 2
            ;;
        --xai-method)
            XAI_METHOD="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --atlas-path)
            ATLAS_PATH="$2"
            shift 2
            ;;
        --individual-patients)
            INDIVIDUAL_PATIENTS="--individual_patients"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

TRAINING_CSV=${TRAINING_CSV:-$DEFAULT_TRAINING_CSV}
TEST_CSVS=${TEST_CSVS:-$DEFAULT_TEST_CSVS}
TRAINING_IMG_DIR=${TRAINING_IMG_DIR:-$DEFAULT_TRAINING_IMG_DIR}
TEST_IMG_DIRS=${TEST_IMG_DIRS:-$DEFAULT_TEST_IMG_DIRS}
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}
MASK_PATHS=${MASK_PATHS:-$DEFAULT_MASK_PATHS}
STATS_OUTPUT_ROOT=${STATS_OUTPUT_ROOT:-$DEFAULT_STATS_OUTPUT_ROOT}
INDICES_PATH=${INDICES_PATH:-$DEFAULT_INDICES_PATH}
XAI_OUTPUT_DIR=${XAI_OUTPUT_DIR:-$DEFAULT_XAI_OUTPUT_DIR}
XAI_METHOD=${XAI_METHOD:-$DEFAULT_XAI_METHOD}
DEVICE=${DEVICE:-$DEFAULT_DEVICE}
ATLAS_PATH=${ATLAS_PATH:-$DEFAULT_ATLAS_PATH}
INDIVIDUAL_PATIENTS=${INDIVIDUAL_PATIENTS:-$DEFAULT_INDIVIDUAL_PATIENTS}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load Anaconda3/2024.02-1
module load cuda/12.8
conda activate brainage

read -a test_csv_array <<< "$TEST_CSVS"
read -a test_img_array <<< "$TEST_IMG_DIRS"
read -a mask_path_array <<< "$MASK_PATHS"

echo "Running statistical analysis..."
python test_stats.py \
    --validation_csv "$TRAINING_CSV" "${test_csv_array[@]}" \
    --validation_img_dir "$TRAINING_IMG_DIR" "${test_img_array[@]}" \
    --model_dir "$MODEL_DIR" \
    --mask_path "${mask_path_array[@]}" \
    --output_root "$STATS_OUTPUT_ROOT" \
    --indices_path "$INDICES_PATH" $(printf "None %.0s" $(seq 1 ${#test_csv_array[@]}))

echo "Running XAI analysis..."
for i in "${!test_csv_array[@]}"; do
    test_csv="${test_csv_array[$i]}"
    test_dir="${test_img_array[$i]}"
    dataset_name=$(basename "$test_csv" .csv)
    
    echo "Processing dataset: $dataset_name"
    python xai.py \
        --models_dir "$MODEL_DIR" \
        --test_csv "$test_csv" \
        --test_data_dir "$test_dir" \
        --output_dir "$XAI_OUTPUT_DIR/$(basename $MODEL_DIR)_$dataset_name" \
        --method "$XAI_METHOD" \
        --device "$DEVICE" \
        --atlas_path "$ATLAS_PATH" \
        $INDIVIDUAL_PATIENTS
done

