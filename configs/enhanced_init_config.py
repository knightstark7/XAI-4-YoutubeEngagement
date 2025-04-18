# Configuration for video association optimization with enhanced initialization

# Basic configuration
project_name = "LaBo2"
data_root = "./data"
work_dir = "./checkpoints"
output_dir = "./results"
video_results_path = "./video_results_cleaned.json"  # Using the cleaned version without missing concepts

# Model settings
num_classes = 2  # engaging and not_engaging
num_concept = 500  # Number of concepts to select
init_val = 1.0  # Value for initializing association weights
asso_act = "softmax"  # Activation function for association matrix: relu, tanh, softmax, or none

# Weight initialization settings
weight_init_method = "llm"  # Options: default, random, continuous, llm
llm_temperature = 0.7  # Temperature parameter for LLM weight scaling

# Embedding settings
embedding_model = "text-embedding-3-large"
use_content_norm = True
use_concept_norm = True

# Concept selection
concept_select_fn = "diversity"  # mi, sim, group_mi, group_sim, diversity, random
use_dot_product = True  # Whether to use dot product data module (faster)

# Training settings
batch_size = 256
lr = 5e-4  # Reduced from 1e-3 to 5e-4 for more stable training
max_epochs = 200
check_val_every_n_epoch = 5
seed = 42
use_l1_loss = True
lambda_l1 = 1e-3
use_div_loss = True
lambda_div = 1e-2

# Data settings
val_ratio = 0.15
test_ratio = 0.15
force_compute = True  # Set to True to force recomputation of embeddings

# System settings
num_workers = 32
on_gpu = True
threshold = 0.5  # Threshold for extracting concepts 