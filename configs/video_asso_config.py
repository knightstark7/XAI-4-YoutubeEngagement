# Configuration for video association optimization

# Basic configuration
project_name = "LaBo2"
data_root = "./data"
work_dir = "./checkpoints"
output_dir = "./results"
video_results_path = "./video_results.json"

# Model settings
num_classes = 2  # engaging and not_engaging
num_concept = 200  # Number of concepts to select
init_val = 1.0  # Value for initializing association weights
asso_act = "softmax"  # Activation function for association matrix: relu, tanh, softmax, or none

# Embedding settings
embedding_model = "text-embedding-3-small"
use_content_norm = True
use_concept_norm = True

# Concept selection
concept_select_fn = "diversity"  # mi, sim, group_mi, group_sim, diversity, random
use_dot_product = True  # Whether to use dot product data module (faster)

# Training settings
batch_size = 128
lr = 1e-2
max_epochs = 100
check_val_every_n_epoch = 5
seed = 42
use_l1_loss = False
lambda_l1 = 1e-3
use_div_loss = False
lambda_div = 1e-2

# Data settings
val_ratio = 0.2
test_ratio = 0.2
force_compute = True  # Set to True to force recomputation of embeddings

# System settings
num_workers = 32
on_gpu = True
threshold = 0.5  # Threshold for extracting concepts 