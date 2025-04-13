import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_embeddings import VideoEmbedding
from utils.utils import pickle_dump, pickle_load, split_data

class VideoContentDataset(Dataset):
    """
    Dataset for video content embeddings and labels
    """
    def __init__(self, content_embeddings, labels, on_gpu=False):
        """
        Initialize the dataset
        
        Args:
            content_embeddings: Video content embeddings
            labels: Video labels
            on_gpu: Whether to load data on GPU
        """
        self.content_embeddings = content_embeddings.cuda() if on_gpu else content_embeddings
        self.labels = labels.cuda() if on_gpu else labels

    def __len__(self):
        """
        Get dataset length
        
        Returns:
            length: Dataset length
        """
        return len(self.content_embeddings)

    def __getitem__(self, idx):
        """
        Get item at index
        
        Args:
            idx: Index
            
        Returns:
            embedding: Content embedding
            label: Label
        """
        return self.content_embeddings[idx], self.labels[idx]

class DotProductDataset(Dataset):
    """
    Dataset with precomputed dot products between content and concept embeddings
    """
    def __init__(self, content_embeddings, concept_embeddings, labels, on_gpu=False):
        """
        Initialize the dataset
        
        Args:
            content_embeddings: Video content embeddings
            concept_embeddings: Concept embeddings
            labels: Video labels
            on_gpu: Whether to load data on GPU
        """
        self.dot_product = (content_embeddings @ concept_embeddings.t())
        self.dot_product = self.dot_product.cuda() if on_gpu else self.dot_product
        self.labels = labels.cuda() if on_gpu else labels

    def __len__(self):
        """
        Get dataset length
        
        Returns:
            length: Dataset length
        """
        return len(self.dot_product)

    def __getitem__(self, idx):
        """
        Get item at index
        
        Args:
            idx: Index
            
        Returns:
            dot_product: Dot product between content and concepts
            label: Label
        """
        return self.dot_product[idx], self.labels[idx]

class VideoDataModule(pl.LightningDataModule):
    """
    Data module for video processing
    """
    def __init__(
            self,
            num_concept,
            data_root,
            concept_select_fn,
            video_results_path,
            batch_size,
            embedding_model="text-embedding-3-small",
            use_content_norm=False,
            use_concept_norm=False,
            num_workers=0,
            on_gpu=False,
            force_compute=False,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42):
        """
        Initialize the data module
        
        Args:
            num_concept: Number of concepts to select
            data_root: Root directory for data
            concept_select_fn: Function for concept selection
            video_results_path: Path to video results JSON
            batch_size: Batch size
            embedding_model: Model for text embeddings
            use_content_norm: Whether to normalize content embeddings
            use_concept_norm: Whether to normalize concept embeddings
            num_workers: Number of workers for data loading
            on_gpu: Whether to load data on GPU
            force_compute: Whether to force computation of embeddings
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            random_state: Random seed
        """
        super().__init__()
        
        # Configuration
        self.num_concept = num_concept
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        self.video_results_path = video_results_path
        self.embedding_model = embedding_model
        self.use_content_norm = use_content_norm
        self.use_concept_norm = use_concept_norm
        self.force_compute = force_compute
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Save paths
        self.content_embeddings_path = self.data_root / 'content_embeddings.pt'
        self.concept_embeddings_path = self.data_root / 'concept_embeddings.pt'
        self.concepts_path = self.data_root / 'concepts.npy'
        self.labels_path = self.data_root / 'labels.pt'
        self.concept2cls_path = self.data_root / 'concept2cls.npy'
        self.class2concepts_path = self.data_root / 'class2concepts.json'
        self.video_ids_path = self.data_root / 'video_ids.npy'
        self.concept_matrix_path = self.data_root / 'concept_matrix.pt'
        self.splits_path = self.data_root / 'splits.pkl'
        self.selected_idx_path = self.data_root / 'selected_idx.pt'
        
        # Get the embeddings
        self.prepare_embeddings()
        
        # Split data
        self.prepare_splits()
        
        # Select concepts
        self.concept_select_fn = concept_select_fn
        self.select_concepts()
        
        # For dataloader
        self.bs = batch_size
        self.num_workers = num_workers
        self.on_gpu = on_gpu
        
    def prepare_embeddings(self):
        """
        Prepare embeddings from video data
        """
        if (not self.content_embeddings_path.exists() or 
            not self.concept_embeddings_path.exists() or 
            self.force_compute):
            print("Extracting video embeddings...")
            
            # Extract embeddings
            video_embedding = VideoEmbedding(embedding_model=self.embedding_model)
            embeddings_data = video_embedding.get_video_embeddings(self.video_results_path)
            
            # Save embeddings and metadata
            torch.save(embeddings_data['content_embeddings'], self.content_embeddings_path)
            torch.save(embeddings_data['concept_embeddings'], self.concept_embeddings_path)
            torch.save(embeddings_data['labels'], self.labels_path)
            torch.save(embeddings_data['concept_matrix'], self.concept_matrix_path)
            np.save(self.concepts_path, np.array(embeddings_data['concepts']))
            np.save(self.concept2cls_path, embeddings_data['concept2cls'].numpy())
            np.save(self.video_ids_path, np.array(embeddings_data['video_ids']))
            
            # Store class to concepts mapping
            with open(self.class2concepts_path, 'w') as f:
                import json
                json.dump(embeddings_data['class2concepts'], f, indent=4)
                
            self.video_ids = embeddings_data['video_ids']
            self.content_embeddings = embeddings_data['content_embeddings']
            self.concept_embeddings = embeddings_data['concept_embeddings']
            self.labels = embeddings_data['labels']
            self.concepts = embeddings_data['concepts']
            self.concept2cls = embeddings_data['concept2cls'].numpy()
            self.concept_matrix = embeddings_data['concept_matrix']
            self.class2concepts = embeddings_data['class2concepts']
        else:
            print("Loading pre-computed embeddings...")
            self.content_embeddings = torch.load(self.content_embeddings_path)
            self.concept_embeddings = torch.load(self.concept_embeddings_path)
            self.labels = torch.load(self.labels_path)
            self.concept_matrix = torch.load(self.concept_matrix_path)
            self.concepts = np.load(self.concepts_path, allow_pickle=True)
            self.concept2cls = np.load(self.concept2cls_path)
            self.video_ids = np.load(self.video_ids_path, allow_pickle=True)
            
            with open(self.class2concepts_path, 'r') as f:
                import json
                self.class2concepts = json.load(f)
        
        # Normalize embeddings if specified
        if self.use_content_norm:
            self.content_embeddings = self.content_embeddings / self.content_embeddings.norm(dim=-1, keepdim=True)
        
        if self.use_concept_norm:
            self.concept_embeddings = self.concept_embeddings / self.concept_embeddings.norm(dim=-1, keepdim=True)
            
    def prepare_splits(self):
        """
        Prepare train/val/test splits
        """
        if not self.splits_path.exists() or self.force_compute:
            print("Creating data splits...")
            video_embeddings = {
                'video_ids': self.video_ids,
                'content_embeddings': self.content_embeddings,
                'labels': self.labels
            }
            self.splits = split_data(
                video_embeddings, 
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                random_state=self.random_state
            )
            pickle_dump(self.splits, self.splits_path)
        else:
            self.splits = pickle_load(self.splits_path)
        
        # Calculate number of videos per class
        self.num_videos_per_class = []
        for cls_label in ['not_engaging', 'engaging']:
            self.num_videos_per_class.append(len(self.splits['train'][cls_label]))
            
    def select_concepts(self):
        """
        Select concepts using the specified selection function
        """
        if not self.selected_idx_path.exists() or self.force_compute:
            print(f"Selecting {self.num_concept} concepts...")
            
            # Get indices for training videos
            train_indices = []
            for cls in ['not_engaging', 'engaging']:
                for video_id, idx in self.splits['train'][cls].items():
                    train_indices.append(idx)
            
            # Get content embeddings and labels for training data
            train_content_embeddings = self.content_embeddings[train_indices]
            train_labels = self.labels[train_indices]
            
            # Select concepts
            self.selected_idx = self.concept_select_fn(
                train_content_embeddings,
                self.concept_embeddings,
                self.concept2cls,
                self.num_concept,
                self.num_videos_per_class
            )
            
            # If we get more concepts than requested, trim to num_concept
            if len(self.selected_idx) > self.num_concept:
                self.selected_idx = self.selected_idx[:self.num_concept]
                
            torch.save(self.selected_idx, self.selected_idx_path)
        else:
            self.selected_idx = torch.load(self.selected_idx_path)
    
    def setup(self, stage=None):
        """
        Set up datasets for each stage
        
        Args:
            stage: Current stage (fit/validate/test/predict)
        """
        # Prepare indices for each split
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Training indices
        for cls in ['not_engaging', 'engaging']:
            for video_id, idx in self.splits['train'][cls].items():
                train_indices.append(idx)
                
        # Validation indices
        for cls in ['not_engaging', 'engaging']:
            for video_id, idx in self.splits['val'][cls].items():
                val_indices.append(idx)
                
        # Test indices
        for cls in ['not_engaging', 'engaging']:
            for video_id, idx in self.splits['test'][cls].items():
                test_indices.append(idx)
        
        # Create datasets
        self.datasets = {
            'train': VideoContentDataset(
                self.content_embeddings[train_indices],
                self.labels[train_indices],
                self.on_gpu
            ),
            'val': VideoContentDataset(
                self.content_embeddings[val_indices],
                self.labels[val_indices],
                self.on_gpu
            ),
            'test': VideoContentDataset(
                self.content_embeddings[test_indices],
                self.labels[test_indices],
                self.on_gpu
            )
        }
        
    def train_dataloader(self):
        """
        Get training dataloader
        
        Returns:
            dataloader: Training dataloader
        """
        return DataLoader(
            self.datasets['train'],
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False
        )
    
    def val_dataloader(self):
        """
        Get validation dataloader
        
        Returns:
            dataloader: Validation dataloader
        """
        return DataLoader(
            self.datasets['val'],
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False
        )
    
    def test_dataloader(self):
        """
        Get test dataloader
        
        Returns:
            dataloader: Test dataloader
        """
        return DataLoader(
            self.datasets['test'],
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False
        )

class VideoDotProductDataModule(VideoDataModule):
    """
    Data module with precomputed dot products between content and concepts
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize with parent class parameters
        """
        super().__init__(*args, **kwargs)
        
    def setup(self, stage=None):
        """
        Set up datasets with dot products
        
        Args:
            stage: Current stage (fit/validate/test/predict)
        """
        # Prepare indices for each split
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Training indices
        for cls in ['not_engaging', 'engaging']:
            for video_id, idx in self.splits['train'][cls].items():
                train_indices.append(idx)
                
        # Validation indices
        for cls in ['not_engaging', 'engaging']:
            for video_id, idx in self.splits['val'][cls].items():
                val_indices.append(idx)
                
        # Test indices
        for cls in ['not_engaging', 'engaging']:
            for video_id, idx in self.splits['test'][cls].items():
                test_indices.append(idx)
                
        # Get selected concept embeddings
        selected_concept_embeddings = self.concept_embeddings[self.selected_idx[:self.num_concept]]
        
        # Create datasets with dot products
        self.datasets = {
            'train': DotProductDataset(
                self.content_embeddings[train_indices],
                selected_concept_embeddings,
                self.labels[train_indices],
                self.on_gpu
            ),
            'val': DotProductDataset(
                self.content_embeddings[val_indices],
                selected_concept_embeddings,
                self.labels[val_indices],
                self.on_gpu
            ),
            'test': DotProductDataset(
                self.content_embeddings[test_indices],
                selected_concept_embeddings,
                self.labels[test_indices],
                self.on_gpu
            )
        } 