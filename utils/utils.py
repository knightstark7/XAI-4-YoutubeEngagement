import os
import torch
import numpy as np
import random
import json
from pathlib import Path
import pickle

def set_seed(seed):
    """
    Set the random seed for reproducibility
    
    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(data, file_path):
    """
    Save data as JSON
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    """
    Load JSON data
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        data: Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def pickle_dump(data, file_path):
    """
    Save data as pickle
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(file_path):
    """
    Load pickle data
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        data: Loaded data
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def split_data(video_embeddings, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split video data into train, validation and test sets
    
    Args:
        video_embeddings: Dictionary containing video embeddings and metadata
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_state: Random seed for reproducibility
        
    Returns:
        splits: Dictionary with train, val, test splits
    """
    set_seed(random_state)
    
    # Extract data
    video_ids = video_embeddings['video_ids']
    content_embeddings = video_embeddings['content_embeddings']
    labels = video_embeddings['labels']
    
    # Get indices for each class
    engaging_indices = [i for i, label in enumerate(labels) if label == 1]
    not_engaging_indices = [i for i, label in enumerate(labels) if label == 0]
    
    # Shuffle indices
    random.shuffle(engaging_indices)
    random.shuffle(not_engaging_indices)
    
    # Calculate split sizes for each class
    n_engaging = len(engaging_indices)
    n_not_engaging = len(not_engaging_indices)
    
    n_val_engaging = int(n_engaging * val_ratio)
    n_test_engaging = int(n_engaging * test_ratio)
    n_train_engaging = n_engaging - n_val_engaging - n_test_engaging
    
    n_val_not_engaging = int(n_not_engaging * val_ratio)
    n_test_not_engaging = int(n_not_engaging * test_ratio)
    n_train_not_engaging = n_not_engaging - n_val_not_engaging - n_test_not_engaging
    
    # Split indices
    train_engaging = engaging_indices[:n_train_engaging]
    val_engaging = engaging_indices[n_train_engaging:n_train_engaging+n_val_engaging]
    test_engaging = engaging_indices[n_train_engaging+n_val_engaging:]
    
    train_not_engaging = not_engaging_indices[:n_train_not_engaging]
    val_not_engaging = not_engaging_indices[n_train_not_engaging:n_train_not_engaging+n_val_not_engaging]
    test_not_engaging = not_engaging_indices[n_train_not_engaging+n_val_not_engaging:]
    
    # Combine indices
    train_indices = train_engaging + train_not_engaging
    val_indices = val_engaging + val_not_engaging
    test_indices = test_engaging + test_not_engaging
    
    # Create class2video dictionaries
    train_class2video = {'engaging': {}, 'not_engaging': {}}
    val_class2video = {'engaging': {}, 'not_engaging': {}}
    test_class2video = {'engaging': {}, 'not_engaging': {}}
    
    # Fill train dictionary
    for idx in train_indices:
        video_id = video_ids[idx]
        label = 'engaging' if labels[idx] == 1 else 'not_engaging'
        train_class2video[label][video_id] = idx
    
    # Fill val dictionary
    for idx in val_indices:
        video_id = video_ids[idx]
        label = 'engaging' if labels[idx] == 1 else 'not_engaging'
        val_class2video[label][video_id] = idx
    
    # Fill test dictionary
    for idx in test_indices:
        video_id = video_ids[idx]
        label = 'engaging' if labels[idx] == 1 else 'not_engaging'
        test_class2video[label][video_id] = idx
    
    # Create splits
    splits = {
        'train': train_class2video,
        'val': val_class2video,
        'test': test_class2video
    }
    
    return splits

def pre_exp(cfg_path, work_dir):
    """
    Prepare for experiment
    
    Args:
        cfg_path: Path to config file
        work_dir: Working directory
        
    Returns:
        cfg: Configuration object
    """
    from utils.config_utils import Config
    
    cfg = Config.fromfile(cfg_path)
    if work_dir is not None:
        cfg.work_dir = work_dir
        os.makedirs(cfg.work_dir, exist_ok=True)
        cfg.dump(os.path.join(cfg.work_dir, os.path.basename(cfg_path)))
    return cfg 