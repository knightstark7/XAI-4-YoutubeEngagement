import torch
import numpy as np
from tqdm import tqdm
import random
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity_score(content_embeddings, concept_embeddings, num_videos_per_class):
    """
    Calculate similarity scores between content and concepts
    
    Args:
        content_embeddings: Video content embeddings
        concept_embeddings: Concept embeddings
        num_videos_per_class: Number of videos per class
        
    Returns:
        scores_mean: Mean similarity scores
    """
    num_cls = len(num_videos_per_class)
    scores_mean = torch.empty((concept_embeddings.shape[0], num_cls))
    start_loc = 0
    for i in range(num_cls):
        end_loc = sum(num_videos_per_class[:i+1])
        sim_matrix = concept_embeddings @ content_embeddings[start_loc:end_loc].t()
        scores_mean[:, i] = sim_matrix.mean(dim=-1)
        start_loc = end_loc
    return scores_mean

def mi_score(content_embeddings, concept_embeddings, num_videos_per_class):
    """
    Calculate mutual information scores
    
    Args:
        content_embeddings: Video content embeddings
        concept_embeddings: Concept embeddings
        num_videos_per_class: Number of videos per class
        
    Returns:
        mi: Mutual information scores
        scores_mean: Mean similarity scores
    """
    num_cls = len(num_videos_per_class)
    scores_mean = get_similarity_score(content_embeddings, concept_embeddings, num_videos_per_class)
    normalized_scores = scores_mean / (scores_mean.sum(dim=0) * num_cls)
    margin_x = normalized_scores.sum(dim=1)
    margin_x = margin_x.reshape(-1, 1).repeat(1, num_cls)
    
    # Compute MI and PMI
    pmi = torch.log(normalized_scores / (margin_x * 1 / num_cls) + 1e-10)
    mi = normalized_scores * pmi
    mi = mi.sum(dim=1)
    return mi, scores_mean

def mi_select(content_embeddings, concept_embeddings, num_videos_per_class, *args):
    """
    Select concepts based on mutual information
    
    Args:
        content_embeddings: Video content embeddings
        concept_embeddings: Concept embeddings
        num_videos_per_class: Number of videos per class
        
    Returns:
        selected_idx: Indices of selected concepts
    """
    mi, _ = mi_score(content_embeddings, concept_embeddings, num_videos_per_class)
    _, selected_idx = torch.sort(mi, descending=True)
    return selected_idx

def sim_score_select(content_embeddings, concept_embeddings, num_videos_per_class, *args):
    """
    Select concepts based on similarity scores
    
    Args:
        content_embeddings: Video content embeddings
        concept_embeddings: Concept embeddings
        num_videos_per_class: Number of videos per class
        
    Returns:
        selected_idx: Indices of selected concepts
    """
    scores_mean = get_similarity_score(content_embeddings, concept_embeddings, num_videos_per_class)
    best_scores_over_cls = scores_mean.max(dim=-1)[0]
    _, selected_idx = torch.sort(best_scores_over_cls, descending=True)
    return selected_idx

def group_mi_select(content_embeddings, concept_embeddings, concept2cls, num_concepts, num_videos_per_class, *args):
    """
    Select concepts based on mutual information, grouped by class
    
    Args:
        content_embeddings: Video content embeddings
        concept_embeddings: Concept embeddings
        concept2cls: Mapping from concepts to classes
        num_concepts: Number of concepts to select
        num_videos_per_class: Number of videos per class
        
    Returns:
        selected_idx: Indices of selected concepts
    """
    assert num_concepts > 0
    num_cls = len(num_videos_per_class)
    scores, _ = mi_score(content_embeddings, concept_embeddings, num_videos_per_class)
    selected_idx = []
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    concept2cls_tensor = torch.from_numpy(concept2cls).long()
    
    for i in tqdm(range(num_cls)):
        cls_idx = torch.where(concept2cls_tensor == i)[0]
        if len(cls_idx) == 0:
            continue
        elif len(cls_idx) < num_concepts_per_cls:
            global_idx = cls_idx
        else:
            _, idx_for_cls_idx = torch.topk(scores[cls_idx], num_concepts_per_cls)
            global_idx = cls_idx[idx_for_cls_idx]
        selected_idx.extend(global_idx)
        
    return torch.tensor(selected_idx)

def group_sim_select(content_embeddings, concept_embeddings, concept2cls, num_concepts, num_videos_per_class, *args):
    """
    Select concepts based on similarity scores, grouped by class
    
    Args:
        content_embeddings: Video content embeddings
        concept_embeddings: Concept embeddings
        concept2cls: Mapping from concepts to classes
        num_concepts: Number of concepts to select
        num_videos_per_class: Number of videos per class
        
    Returns:
        selected_idx: Indices of selected concepts
    """
    assert num_concepts > 0
    num_cls = len(num_videos_per_class)
    scores = get_similarity_score(content_embeddings, concept_embeddings, num_videos_per_class).max(dim=-1)[0]
    selected_idx = []
    concept2cls_tensor = torch.from_numpy(concept2cls).long()
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    
    for i in tqdm(range(num_cls)):
        cls_idx = torch.where(concept2cls_tensor == i)[0]
        if len(cls_idx) == 0:
            continue
        elif len(cls_idx) < num_concepts_per_cls:
            global_idx = cls_idx
        else:
            _, idx_for_cls_idx = torch.topk(scores[cls_idx], num_concepts_per_cls)
            global_idx = cls_idx[idx_for_cls_idx]
        selected_idx.extend(global_idx)
        
    return torch.tensor(selected_idx)

def diversity_select(content_embeddings, concept_embeddings, concept2cls, num_concepts, num_videos_per_class, *args):
    """
    Select diverse concepts using a greedy approach
    
    Args:
        content_embeddings: Video content embeddings
        concept_embeddings: Concept embeddings
        concept2cls: Mapping from concepts to classes
        num_concepts: Number of concepts to select
        num_videos_per_class: Number of videos per class
        
    Returns:
        selected_idx: Indices of selected concepts
    """
    assert num_concepts > 0
    num_cls = len(num_videos_per_class)
    concept2cls_tensor = torch.from_numpy(concept2cls).long()
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    selected_idx = []
    
    # Calculate initial relevance scores
    scores, _ = mi_score(content_embeddings, concept_embeddings, num_videos_per_class)
    
    # For each class, select diverse concepts
    for i in tqdm(range(num_cls)):
        cls_idx = torch.where(concept2cls_tensor == i)[0]
        if len(cls_idx) == 0:
            continue
        elif len(cls_idx) <= num_concepts_per_cls:
            selected_idx.extend(cls_idx)
        else:
            # Calculate similarity between concepts
            selected_concepts_for_cls = []
            cls_concept_embeddings = concept_embeddings[cls_idx]
            cls_scores = scores[cls_idx]
            
            # Start with highest scoring concept
            best_idx = torch.argmax(cls_scores).item()
            selected_concepts_for_cls.append(best_idx)
            
            # Greedily select remaining concepts
            while len(selected_concepts_for_cls) < num_concepts_per_cls:
                # Calculate similarity to already selected concepts
                selected_embeds = cls_concept_embeddings[selected_concepts_for_cls]
                sim_to_selected = torch.matmul(cls_concept_embeddings, selected_embeds.t())
                max_sim_to_selected, _ = torch.max(sim_to_selected, dim=1)
                
                # Adjust scores by similarity (penalize similar concepts)
                adjusted_scores = cls_scores - 0.5 * max_sim_to_selected
                
                # Mask already selected concepts
                mask = torch.ones(len(cls_idx), dtype=torch.bool)
                mask[selected_concepts_for_cls] = False
                adjusted_scores[~mask] = float('-inf')
                
                # Select the next best concept
                next_best = torch.argmax(adjusted_scores).item()
                selected_concepts_for_cls.append(next_best)
            
            # Map back to global indices
            global_idx = [cls_idx[i] for i in selected_concepts_for_cls]
            selected_idx.extend(global_idx)
            
    return torch.tensor(selected_idx)

def random_select(content_embeddings, concept_embeddings, concept2cls, num_concepts, num_videos_per_class, *args):
    """
    Randomly select concepts
    
    Args:
        content_embeddings: Video content embeddings
        concept_embeddings: Concept embeddings
        concept2cls: Mapping from concepts to classes
        num_concepts: Number of concepts to select
        num_videos_per_class: Number of videos per class
        
    Returns:
        selected_idx: Indices of selected concepts
    """
    assert num_concepts > 0
    num_cls = len(num_videos_per_class)
    concept2cls_tensor = torch.from_numpy(concept2cls).long()
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    selected_idx = []
    
    for i in tqdm(range(num_cls)):
        cls_idx = torch.where(concept2cls_tensor == i)[0]
        if len(cls_idx) == 0:
            continue
        elif len(cls_idx) <= num_concepts_per_cls:
            global_idx = cls_idx
        else:
            global_idx = torch.tensor(random.sample(cls_idx.tolist(), num_concepts_per_cls))
        selected_idx.extend(global_idx)
        
    return torch.tensor(selected_idx) 