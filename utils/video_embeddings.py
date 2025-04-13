import torch
import numpy as np
import json
from openai import OpenAI
from tqdm import tqdm

class VideoEmbedding:
    """
    Extract embeddings from video content and audio_tags
    """
    def __init__(self, embedding_model="text-embedding-3-small"):
        """
        Initialize with embedding model
        
        Args:
            embedding_model: Model to use for text embeddings
        """
        self.client = OpenAI()
        self.embedding_model = embedding_model
        
    def get_embeddings(self, texts, batch_size=16):
        """
        Get embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for API calls
            
        Returns:
            embeddings: Numpy array of embeddings
        """
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error getting embeddings for batch {i}: {str(e)}")
                # Add empty embeddings for failed batches
                embeddings.extend([np.zeros(1536) for _ in range(len(batch))])
                
        return np.array(embeddings)
    
    def extract_from_video_results(self, video_results_path):
        """
        Extract content and audio tags from video_results.json
        
        Args:
            video_results_path: Path to video_results.json
            
        Returns:
            video_ids: List of video IDs
            contents: List of text contents
            labels: List of labels (0 for not_engaging, 1 for engaging)
            concepts_by_video: Dictionary mapping video IDs to their concepts
        """
        try:
            # Mở file với encoding UTF-8
            with open(video_results_path, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
        except UnicodeDecodeError:
            # Nếu UTF-8 không hoạt động, thử với các encoding khác
            try:
                with open(video_results_path, 'r', encoding='utf-16') as f:
                    video_data = json.load(f)
            except:
                try:
                    # Tùy chọn cuối cùng: latin-1 (sẽ không bị lỗi nhưng có thể hiển thị sai các ký tự đặc biệt)
                    with open(video_results_path, 'r', encoding='latin-1') as f:
                        video_data = json.load(f)
                except Exception as e:
                    print(f"Không thể đọc file do lỗi encoding: {str(e)}")
                    raise
        
        video_ids = []
        contents = []
        labels = []
        concepts_by_video = {}
        
        for video_id, video_info in video_data.items():
            video_ids.append(video_id)
            
            # Combine content and audio_tags
            content_text = video_info.get('content', '')
            audio_tags = video_info.get('audio_tags', '')
            combined_text = f"{content_text} {audio_tags}"
            contents.append(combined_text)
            
            # Extract label (0 for not_engaging, 1 for engaging)
            label = 1 if video_info.get('label') == 'engaging' else 0
            labels.append(label)
            
            # Store concepts for each video
            concepts_by_video[video_id] = video_info.get('concepts', [])
        
        return video_ids, contents, labels, concepts_by_video
    
    def get_video_embeddings(self, video_results_path):
        """
        Extract and embed video contents
        
        Args:
            video_results_path: Path to video_results.json
            
        Returns:
            video_embeddings: Dictionary with video embeddings and metadata
        """
        video_ids, contents, labels, concepts_by_video = self.extract_from_video_results(video_results_path)
        
        # Get embeddings for contents
        content_embeddings = self.get_embeddings(contents)
        
        # Create list of all unique concepts
        all_concepts = set()
        for concepts in concepts_by_video.values():
            all_concepts.update(concepts)
        all_concepts = list(all_concepts)
        
        # Map concepts to classes
        concept2cls = np.zeros(len(all_concepts), dtype=int)
        class2concepts = {0: [], 1: []}
        
        for i, concept in enumerate(all_concepts):
            # Check which class (engaging/not_engaging) uses this concept more
            count_engaging = sum(1 for vid_id in video_ids 
                               if concept in concepts_by_video[vid_id] 
                               and labels[video_ids.index(vid_id)] == 1)
            count_not_engaging = sum(1 for vid_id in video_ids 
                                    if concept in concepts_by_video[vid_id] 
                                    and labels[video_ids.index(vid_id)] == 0)
            
            # Assign concept to the class that uses it more
            assigned_class = 1 if count_engaging > count_not_engaging else 0
            concept2cls[i] = assigned_class
            class2concepts[assigned_class].append(concept)
        
        # Get embeddings for all concepts
        concept_embeddings = self.get_embeddings(all_concepts)
        
        # Create binary concept matrix (videos x concepts)
        concept_matrix = np.zeros((len(video_ids), len(all_concepts)))
        for i, vid_id in enumerate(video_ids):
            for j, concept in enumerate(all_concepts):
                if concept in concepts_by_video[vid_id]:
                    concept_matrix[i, j] = 1
        
        # Convert to PyTorch tensors
        content_embeddings_tensor = torch.tensor(content_embeddings, dtype=torch.float32)
        concept_embeddings_tensor = torch.tensor(concept_embeddings, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        concept_matrix_tensor = torch.tensor(concept_matrix, dtype=torch.float32)
        concept2cls_tensor = torch.tensor(concept2cls, dtype=torch.long)
        
        return {
            'video_ids': video_ids,
            'content_embeddings': content_embeddings_tensor,
            'concept_embeddings': concept_embeddings_tensor,
            'labels': labels_tensor,
            'concept_matrix': concept_matrix_tensor,
            'concepts': all_concepts,
            'concept2cls': concept2cls_tensor,
            'class2concepts': class2concepts
        } 