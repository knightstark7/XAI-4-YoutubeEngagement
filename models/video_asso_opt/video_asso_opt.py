import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

class VideoAssoConcept(pl.LightningModule):
    """
    Model for video association concept optimization
    """
    def __init__(self, cfg, init_weight=None, select_idx=None):
        """
        Initialize the model
        
        Args:
            cfg: Configuration object
            init_weight: Initial weights for the association matrix
            select_idx: Indices of selected concepts
        """
        super().__init__()
        self.cfg = cfg
        self.concept_feat_path = f"{cfg.data_root}/concept_embeddings.pt"
        self.selected_idx_path = f"{cfg.data_root}/selected_idx.pt"
        self.concepts_path = f"{cfg.data_root}/concepts.npy"
        self.concept2cls_path = f"{cfg.data_root}/concept2cls.npy"
        
        # Load concept embeddings and metadata
        if select_idx is None:
            self.select_idx = torch.load(self.selected_idx_path)[:cfg.num_concept]
        else:
            self.select_idx = select_idx
            
        # Load concept embeddings
        self.register_buffer('concepts', torch.load(self.concept_feat_path)[self.select_idx])
        if hasattr(cfg, 'use_concept_norm') and cfg.use_concept_norm:
            self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)
            
        # Load concept names and class associations
        self.concept_raw = np.load(self.concepts_path, allow_pickle=True)[self.select_idx]
        self.register_buffer('concept2cls', torch.from_numpy(np.load(self.concept2cls_path))[self.select_idx].long().view(1, -1))
        
        # Initialize association matrix
        if init_weight is None:
            self.init_weight_concept(self.concept2cls)
        else:
            self.init_weight = init_weight
            
        # Create association matrix as learnable parameter
        self.asso_mat = nn.Parameter(self.init_weight.clone())
        
        # Set up metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=cfg.num_classes)
        
        # For test results
        self.all_y = []
        self.all_pred = []
        
        # Save hyperparameters
        self.save_hyperparameters()
        
    def init_weight_concept(self, concept2cls):
        """
        Initialize weights for the association matrix
        
        Args:
            concept2cls: Mapping from concepts to classes
        """
        # Initialize with zeros
        self.init_weight = torch.zeros((self.cfg.num_classes, len(self.select_idx)), device=self.device)
        
        # Use random initialization if specified
        if hasattr(self.cfg, 'use_rand_init') and self.cfg.use_rand_init:
            nn.init.kaiming_normal_(self.init_weight)
        else:
            # Initialize with one-hot associations based on concept2cls
            self.init_weight.scatter_(0, concept2cls, self.cfg.init_val)
    
    def _get_weight_mat(self):
        """
        Apply activation function to association matrix
        
        Returns:
            mat: Processed association matrix
        """
        if self.cfg.asso_act == 'relu':
            mat = F.relu(self.asso_mat)
        elif self.cfg.asso_act == 'tanh':
            mat = F.tanh(self.asso_mat)
        elif self.cfg.asso_act == 'softmax':
            mat = F.softmax(self.asso_mat, dim=0)
        else:
            mat = self.asso_mat
        return mat
    
    def forward(self, content_feat):
        """
        Forward pass
        
        Args:
            content_feat: Video content embeddings
            
        Returns:
            sim: Similarity scores between content and classes
        """
        # Get processed association matrix
        mat = self._get_weight_mat()
        
        # Move content_feat to same device as model if needed
        if content_feat.device != self.device:
            content_feat = content_feat.to(self.device)
        
        # Compute class embeddings as weighted sum of concept embeddings
        cls_feat = mat @ self.concepts
        
        # Compute similarity between content and class embeddings
        sim = content_feat @ cls_feat.t()
        
        return sim
    
    def save_predictions(self, phase, pred_labels, actual_labels, limit=100):
        """
        Save predictions and actual labels to a formatted CSV file for easy viewing
        
        Args:
            phase: Training phase (train, val, test)
            pred_labels: Predicted labels
            actual_labels: Actual labels
            limit: Maximum number of samples to save
        """
        if not hasattr(self.cfg, 'output_dir'):
            return
            
        # Ensure output directory exists
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        # Create data for CSV
        num_samples = min(limit, len(pred_labels))
        data = []
        
        for i in range(num_samples):
            pred = pred_labels[i].item() if isinstance(pred_labels[i], torch.Tensor) else pred_labels[i]
            actual = actual_labels[i].item() if isinstance(actual_labels[i], torch.Tensor) else actual_labels[i]
            
            # Convert numeric labels to text labels
            pred_text = "engaging" if pred == 1 else "not_engaging"
            actual_text = "engaging" if actual == 1 else "not_engaging"
            
            # Check if prediction is correct
            is_correct = pred == actual
            
            data.append([i, pred, pred_text, actual, actual_text, is_correct])
        
        # Calculate accuracy
        correct_count = sum(1 for _, _, _, _, _, is_correct in data if is_correct)
        accuracy = correct_count / num_samples if num_samples > 0 else 0
        
        # Save to CSV
        import csv
        output_path = os.path.join(self.cfg.output_dir, f"{phase}_predictions.csv")
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Sample', 'Predicted (Numeric)', 'Predicted (Text)', 
                             'Actual (Numeric)', 'Actual (Text)', 'Is Correct'])
            
            for row in data:
                writer.writerow(row)
                
            # Add summary row
            writer.writerow([''])
            writer.writerow(['Summary:', f'Accuracy: {accuracy:.4f}', 
                           f'Correct: {correct_count}/{num_samples}'])
        
        print(f"{phase.capitalize()} predictions saved to {output_path}")
        
        # Save as formatted JSON for additional use
        json_data = {
            "predictions": [{"sample": i, 
                           "predicted": {"numeric": p[1], "text": p[2]}, 
                           "actual": {"numeric": p[3], "text": p[4]}, 
                           "is_correct": p[5]} 
                           for i, p in enumerate(data)],
            "summary": {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_samples": num_samples
            }
        }
        
        json_path = os.path.join(self.cfg.output_dir, f"{phase}_predictions.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
    
    def training_step(self, train_batch, batch_idx):
        """
        Training step
        
        Args:
            train_batch: Batch of training data
            batch_idx: Batch index
            
        Returns:
            loss: Training loss
        """
        content_embed, label = train_batch
        
        # Forward pass
        sim = self.forward(content_embed)
        pred = 100 * sim  # Scaling as in standard CLIP
        
        # Classification loss
        cls_loss = F.cross_entropy(pred, label)
        
        # Diverse response regularization
        div = -torch.var(sim, dim=0).mean()
        
        # L1 regularization for sparsity
        row_l1_norm = torch.linalg.vector_norm(self.asso_mat, ord=1, dim=-1).max()
        
        # Get predicted labels
        pred_labels = torch.argmax(pred, dim=1)
        
        # Lưu predicted và actual labels trong batch để xử lý trong on_train_epoch_end
        if not hasattr(self, 'train_pred_labels'):
            self.train_pred_labels = []
            self.train_actual_labels = []
        
        self.train_pred_labels.append(pred_labels.detach().cpu())
        self.train_actual_labels.append(label.detach().cpu())
        
        # Log predicted and actual labels for first few examples in batch
        for i in range(min(5, len(label))):
            self.log(f'train_pred_label_{i}', pred_labels[i].item(), on_step=True, on_epoch=False)
            self.log(f'train_actual_label_{i}', label[i].item(), on_step=True, on_epoch=False)
        
        # Log metrics cho mỗi batch
        self.log('train_loss_step', cls_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('mean_l1_norm', row_l1_norm, on_step=True, on_epoch=True)
        self.log('div', div, on_step=True, on_epoch=True)
        
        # Update accuracy
        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute final loss with regularization
        final_loss = cls_loss
        if hasattr(self.cfg, 'use_l1_loss') and self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if hasattr(self.cfg, 'use_div_loss') and self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        
        # Log final loss
        self.log('training_loss', final_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Lưu loss để sử dụng trong on_train_epoch_end
        self.last_loss = final_loss
        
        return {"loss": final_loss, "pred_labels": pred_labels, "actual_labels": label}
    
    def on_train_epoch_end(self):
        """
        Called at the end of training epoch
        """
        # Log epoch-level metrics
        self.log('train_loss_epoch', self.last_loss, prog_bar=True)
        
        # Lưu predictions và actual labels cho epoch hiện tại
        if hasattr(self, 'train_pred_labels') and len(self.train_pred_labels) > 0:
            all_pred = torch.cat(self.train_pred_labels)
            all_actual = torch.cat(self.train_actual_labels)
            
            # Lưu vào file để dễ theo dõi
            self.save_predictions('train', all_pred, all_actual)
            
            # Reset cho epoch tiếp theo
            self.train_pred_labels = []
            self.train_actual_labels = []
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Batch of validation data
            batch_idx: Batch index
            
        Returns:
            loss: Validation loss
        """
        content_embed, label = batch
        
        # Forward pass
        sim = self.forward(content_embed)
        pred = 100 * sim
        
        # Get predicted labels
        pred_labels = torch.argmax(pred, dim=1)
        
        # Lưu predicted và actual labels trong batch để xử lý trong on_validation_epoch_end
        if not hasattr(self, 'val_pred_labels'):
            self.val_pred_labels = []
            self.val_actual_labels = []
        
        self.val_pred_labels.append(pred_labels.detach().cpu())
        self.val_actual_labels.append(label.detach().cpu())
        
        # Log predicted and actual labels for first few examples in batch
        for i in range(min(5, len(label))):
            self.log(f'val_pred_label_{i}', pred_labels[i].item(), on_step=False, on_epoch=True)
            self.log(f'val_actual_label_{i}', label[i].item(), on_step=False, on_epoch=True)
        
        # Compute loss
        loss = F.cross_entropy(pred, label)
        
        # Log metrics
        self.log('val_loss', loss)
        self.valid_acc(pred, label)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        
        return {"loss": loss, "pred_labels": pred_labels, "actual_labels": label}
    
    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch
        """
        # Lưu predictions và actual labels cho epoch hiện tại
        if hasattr(self, 'val_pred_labels') and len(self.val_pred_labels) > 0:
            all_pred = torch.cat(self.val_pred_labels)
            all_actual = torch.cat(self.val_actual_labels)
            
            # Lưu vào file để dễ theo dõi
            self.save_predictions('val', all_pred, all_actual)
            
            # Reset cho epoch tiếp theo
            self.val_pred_labels = []
            self.val_actual_labels = []
    
    def test_step(self, batch, batch_idx):
        """
        Test step
        
        Args:
            batch: Batch of test data
            batch_idx: Batch index
            
        Returns:
            loss: Test loss
        """
        content_embed, label = batch
        
        # Forward pass
        sim = self.forward(content_embed)
        pred = 100 * sim
        
        # Get predicted labels
        pred_labels = torch.argmax(pred, dim=1)
        
        # Log predicted and actual labels for first few examples in batch
        for i in range(min(5, len(label))):
            self.log(f'test_pred_label_{i}', pred_labels[i].item(), on_step=False, on_epoch=True)
            self.log(f'test_actual_label_{i}', label[i].item(), on_step=False, on_epoch=True)
        
        # Compute loss and update metrics
        loss = F.cross_entropy(pred, label)
        self.log('test_loss', loss)
        self.test_acc(pred, label)
        self.confmat(pred, label)
        
        # Store predictions and labels for later analysis
        self.all_y.append(label)
        self.all_pred.append(pred_labels)
        
        return {"loss": loss, "pred_labels": pred_labels, "actual_labels": label}
    
    def on_test_epoch_end(self):
        """
        End of test epoch actions
        """
        # Concatenate all predictions and labels
        all_y = torch.cat(self.all_y)
        all_pred = torch.cat(self.all_pred)
        
        # Compute total accuracy
        self.total_test_acc = self.test_acc.compute()
        
        # Lưu kết quả dự đoán vào file CSV và JSON dễ đọc
        self.save_predictions('test', all_pred, all_y, limit=len(all_pred))
        
        # Thay vì log string, chúng ta sẽ log số lượng nhãn mỗi loại
        # và lưu predictions vào file JSON
        if hasattr(self.cfg, 'output_dir'):
            # Đếm số lượng nhãn dự đoán mỗi loại (engaging/not_engaging)
            pred_count = torch.bincount(all_pred, minlength=2)
            self.log('pred_count_not_engaging', pred_count[0].item())
            self.log('pred_count_engaging', pred_count[1].item())
            
            # Đếm số lượng nhãn thực tế mỗi loại
            actual_count = torch.bincount(all_y, minlength=2)
            self.log('actual_count_not_engaging', actual_count[0].item())
            self.log('actual_count_engaging', actual_count[1].item())
            
            # Lưu kết quả dự đoán vào file JSON chi tiết hơn
            results_dict = {
                "summary": {
                    "accuracy": self.total_test_acc.item(),
                    "not_engaging_count": {
                        "actual": actual_count[0].item(),
                        "predicted": pred_count[0].item()
                    },
                    "engaging_count": {
                        "actual": actual_count[1].item(), 
                        "predicted": pred_count[1].item()
                    }
                }
            }
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            
            # Lưu vào file
            with open(os.path.join(self.cfg.output_dir, 'test_predictions_summary.json'), 'w') as f:
                json.dump(results_dict, f, indent=4)
        
        # Get confusion matrix
        conf_matrix = self.confmat.compute().cpu().numpy()
        
        # Calculate per-class metrics
        class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        # Log metrics
        self.log('test_acc', self.total_test_acc)
        self.log('not_engaging_acc', class_acc[0])
        self.log('engaging_acc', class_acc[1])
        
        # Save confusion matrix visualization if output directory is available
        if hasattr(self.cfg, 'output_dir'):
            try:
                self.plot_confusion_matrix(conf_matrix, 
                                          ['not_engaging', 'engaging'],
                                          self.cfg.output_dir)
            except Exception as e:
                print(f"Warning: Could not save confusion matrix plot: {e}")
        
        # Clear stored data
        self.all_y = []
        self.all_pred = []
    
    def plot_confusion_matrix(self, cm, class_names, output_dir):
        """
        Plot confusion matrix and save to file
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            output_dir: Directory to save the plot
        """
        output_path = Path(output_dir) / 'confusion_matrix.png'
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2%})",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.savefig(output_path)
        plt.close(figure)
        
        print(f"Confusion matrix plot saved to {output_path}")
    
    def configure_optimizers(self):
        """
        Configure optimizer
        
        Returns:
            optimizer: Optimizer for training
        """
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    def prune_asso_mat(self, q=0.9, thresh=None):
        """
        Prune association matrix by removing weak associations
        
        Args:
            q: Quantile threshold
            thresh: Absolute threshold (overrides q if provided)
            
        Returns:
            good: Boolean mask of kept associations
        """
        asso_mat = self._get_weight_mat().detach()
        val_asso_mat = torch.abs(asso_mat).max(dim=0)[0]
        
        if thresh is None:
            thresh = torch.quantile(val_asso_mat, q)
            
        good = val_asso_mat >= thresh
        return good
    
    def extract_cls2concept(self, thresh=0.05):
        """
        Extract class to concept associations
        
        Args:
            thresh: Threshold for association strength
            
        Returns:
            res: Dictionary mapping class names to associated concepts
        """
        asso_mat = self._get_weight_mat().detach()
        strong_asso = asso_mat > thresh
        
        res = {}
        for i in range(asso_mat.shape[0]):
            # Get indices of concepts strongly associated with class i
            keep_idx = strong_asso[i]
            
            # Map to concept names
            class_name = "engaging" if i == 1 else "not_engaging"
            res[class_name] = self.concept_raw[keep_idx.cpu().numpy()].tolist()
            
        return res
    
    def extract_concept2cls(self, percent_thresh=0.95):
        """
        Extract concept to class associations
        
        Args:
            percent_thresh: Percentile threshold
            
        Returns:
            res: Dictionary mapping concept indices to class rankings
        """
        asso_mat = self.asso_mat.detach()
        res = {}
        
        for i in range(asso_mat.shape[1]):
            # Sort classes by association strength in descending order
            res[i] = torch.argsort(asso_mat[:, i], descending=True).tolist()
            
        return res

class VideoAssoConceptFast(VideoAssoConcept):
    """
    Faster version that works with precomputed dot products
    """
    def forward(self, dot_product):
        """
        Forward pass with precomputed dot products
        
        Args:
            dot_product: Precomputed dot products between content and concepts
            
        Returns:
            sim: Similarity scores
        """
        mat = self._get_weight_mat()
        return dot_product @ mat.t() 