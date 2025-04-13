#!/bin/bash

# Set OpenAI API key
echo "Enter your OpenAI API key:"
read OPENAI_API_KEY
export OPENAI_API_KEY=$OPENAI_API_KEY

# Setting up configuration
CONCEPT_SELECT_METHOD="diversity"  # Should match the trained model
NUM_CONCEPTS=40
EXPERIMENT_NAME="${CONCEPT_SELECT_METHOD}_${NUM_CONCEPTS}concepts"
CHECKPOINT_PATH="./runs/${EXPERIMENT_NAME}"  # Directory containing the checkpoint
RESULTS_DIR="./results/${EXPERIMENT_NAME}"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found at $CHECKPOINT_PATH"
    exit 1
fi

# Find the best checkpoint (highest val_acc)
BEST_CHECKPOINT=$(find $CHECKPOINT_PATH -name "*.ckpt" | sort -r | head -n 1)

if [ -z "$BEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $CHECKPOINT_PATH"
    exit 1
fi

echo "Using checkpoint: $BEST_CHECKPOINT"

# Create results directory
mkdir -p $RESULTS_DIR

# Run testing
python main.py \
    --cfg configs/video_asso_config.py \
    --test \
    --cfg-options \
        ckpt_path="$BEST_CHECKPOINT" \
        video_results_path="./video_results.json" \
        concept_select_fn=$CONCEPT_SELECT_METHOD \
        num_concept=$NUM_CONCEPTS \
        data_root="./data/${EXPERIMENT_NAME}" \
        output_dir=$RESULTS_DIR

echo "Testing complete! Results saved to $RESULTS_DIR"
echo "Optimized concepts saved to $RESULTS_DIR/optimized_concepts.json" 