#!/bin/bash

# Set OpenAI API key
echo "Enter your OpenAI API key:"
read OPENAI_API_KEY
export OPENAI_API_KEY=$OPENAI_API_KEY

# Setting up configuration
CONCEPT_SELECT_METHOD="diversity"  # Options: mi, sim, group_mi, group_sim, diversity, random
NUM_CONCEPTS=40
EXPERIMENT_NAME="${CONCEPT_SELECT_METHOD}_${NUM_CONCEPTS}concepts"
WORK_DIR="./runs/${EXPERIMENT_NAME}"

# Create directories
mkdir -p $WORK_DIR

# Run training
python main.py \
    --cfg configs/video_asso_config.py \
    --work-dir $WORK_DIR \
    --cfg-options \
        video_results_path="./video_results.json" \
        concept_select_fn=$CONCEPT_SELECT_METHOD \
        num_concept=$NUM_CONCEPTS \
        data_root="./data/${EXPERIMENT_NAME}" \
        output_dir="./results/${EXPERIMENT_NAME}"

echo "Training complete! Checkpoints saved to $WORK_DIR" 