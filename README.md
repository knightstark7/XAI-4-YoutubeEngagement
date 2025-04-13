# LaBo2: Video Classification with Concept Association Optimization

LaBo2 extends the LaBo (LAnguage-guided concept BOttlenecks) framework to video classification, specifically for determining whether videos are "engaging" or "not engaging". The framework uses language concepts (text descriptions) as an interpretable bottleneck for classification.

## Overview

LaBo2 works by:
1. Extracting embeddings from video content (text descriptions and audio tags)
2. Selecting relevant concepts associated with each class
3. Learning an association matrix between concepts and classes
4. Using the optimized associations to make classifications

The framework is highly interpretable as it provides clear concepts associated with each class.

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch
- PyTorch Lightning
- OpenAI (for text embeddings)
- NumPy
- tqdm
- scikit-learn
- matplotlib (for visualizations)

## Project Structure

```
LaBo2/
├── configs/                 # Configuration files
│   └── video_asso_config.py # Sample configuration 
├── data/                    # Data directory (created during execution)
├── models/                  # Model implementations
│   ├── select_concept/      # Concept selection algorithms
│   └── video_asso_opt/      # Video association optimization models
├── utils/                   # Utility functions
├── main.py                  # Main execution script
├── README.md                # This file
└── requirements.txt         # Required packages
```

## How It Works

### 1. Data Preparation

The system processes video data from a JSON file (`video_results.json`) containing:
- Video ID
- Label (engaging/not_engaging)
- Content (text description of the video)
- Audio tags (audio information)
- Concepts (associated concepts for each video)

Example format:
```json
{
  "video_id_1": {
    "label": "engaging",
    "content": "Scene 1 - Visual: a group of people dancing...",
    "audio_tags": "Music, Speech",
    "concepts": [
      "Dynamic movement and action",
      "Energetic music and sounds",
      "Celebration atmosphere"
    ]
  },
  ...
}
```

### 2. Embedding Extraction

The system extracts embeddings from:
- Combined content and audio tags (using OpenAI's embedding models)
- Concepts listed for each video

The embeddings are saved to disk for reuse in subsequent runs.

### 3. Concept Selection

Various methods for selecting the most relevant concepts:
- Mutual Information (MI)
- Similarity Score
- Group-based selection (by class)
- Diversity-based selection

### 4. Association Optimization

The system learns an association matrix that maps concepts to classes. This is the core of the LaBo framework. The optimization process uses:
- Cross-entropy loss for classification
- L1 regularization for sparsity
- Diversity loss to encourage varied concept usage

### 5. Evaluation and Interpretation

After training, the system:
- Evaluates classification performance on the test set
- Extracts the optimized concept-class associations
- Outputs the most relevant concepts for each class
- Saves confusion matrix and performance metrics

## Usage

### Training

To train a model:

```bash
python main.py --cfg configs/video_asso_config.py --work-dir ./runs/experiment1
```

Options:
- `--cfg`: Path to configuration file
- `--work-dir`: Directory to save checkpoints and logs
- `--DEBUG`: Enable debug mode
- `--cfg-options`: Override configuration options

### Testing

To test a trained model:

```bash
python main.py --cfg configs/video_asso_config.py --test --cfg-options ckpt_path=./runs/experiment1/epoch=50-step=1000-val_acc=0.8500.ckpt output_dir=./results/experiment1
```

This will:
1. Load the checkpoint
2. Evaluate on the test set
3. Extract optimized concepts
4. Save results to the specified output directory

## Logging and Monitoring

The framework uses PyTorch Lightning's CSV Logger to record metrics during training and validation. All metrics are saved to CSV files in the working directory, making it easy to analyze results after training.

After training completes:
- Training metrics are saved in CSV format
- A confusion matrix visualization is saved as PNG
- Optimized concepts are saved as JSON
- Experiment configuration is saved for reproducibility

You can find these files in the specified working directory and output directory.

## Configuration Options

Key configuration options:

| Option | Description |
|--------|-------------|
| `data_root` | Root directory for data storage |
| `video_results_path` | Path to video results JSON |
| `num_concept` | Number of concepts to select |
| `concept_select_fn` | Concept selection method (mi, sim, group_mi, group_sim, diversity, random) |
| `embedding_model` | Model for text embeddings (OpenAI) |
| `use_dot_product` | Whether to use dot product data module (faster training) |
| `asso_act` | Activation function for association matrix (relu, tanh, softmax, none) |
| `use_l1_loss` | Whether to use L1 regularization |
| `use_div_loss` | Whether to use diversity loss |
| `force_compute` | Force recomputation of embeddings |

## Flow Diagram

```
Video Data (JSON) → Embeddings → Concept Selection → Association Optimization → Evaluation → Concept Extraction
```

## Examples

### Training with Different Concept Selection Methods

```bash
# Using diversity-based selection
python main.py --cfg configs/video_asso_config.py --cfg-options concept_select_fn=diversity num_concept=40

# Using mutual information
python main.py --cfg configs/video_asso_config.py --cfg-options concept_select_fn=mi num_concept=40

# Using group-based selection
python main.py --cfg configs/video_asso_config.py --cfg-options concept_select_fn=group_sim num_concept=40
```

### Testing with Different Thresholds

```bash
# Test with standard threshold
python main.py --cfg configs/video_asso_config.py --test --cfg-options ckpt_path=./path/to/checkpoint.ckpt

python main.py --cfg configs/video_asso_config.py --test --cfg-options ckpt_path=D:/school/Thesis/LaBo2/checkpoints/epoch=4-step=10-val_acc=0.5000.ckpt

# Test with stricter threshold (fewer concepts)
python main.py --cfg configs/video_asso_config.py --test --cfg-options ckpt_path=./path/to/checkpoint.ckpt threshold=0.1

# Test with looser threshold (more concepts)
python main.py --cfg configs/video_asso_config.py --test --cfg-options ckpt_path=./path/to/checkpoint.ckpt threshold=0.01
```

## References

This implementation is based on the LaBo framework:
- [LaBo: Language-guided concept BOttlenecks](https://github.com/YujieLu10/LaBo) 