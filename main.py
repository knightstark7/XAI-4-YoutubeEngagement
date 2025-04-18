import os
import argparse
import pytorch_lightning as pl
import torch
import numpy as np
import random
import json
from pathlib import Path

from models.video_asso_opt.video_asso_opt import VideoAssoConcept, VideoAssoConceptFast
# Import mô hình cải tiến
from models.video_asso_opt.video_asso_concept_enhanced import VideoAssoConceptEnhanced, VideoAssoConceptFastEnhanced
from models.select_concept.select_video_concepts import (
    mi_select, 
    sim_score_select, 
    group_mi_select, 
    group_sim_select,
    diversity_select,
    random_select
)
from data.video_data import VideoDataModule, VideoDotProductDataModule
from utils.utils import pre_exp, set_seed
from utils.config_utils import Config, DictAction

def setup_concept_select_fn(cfg):
    """
    Set up the concept selection function based on configuration
    
    Args:
        cfg: Configuration object
        
    Returns:
        concept_select_fn: Function for concept selection
    """
    if cfg.concept_select_fn == "mi":
        return mi_select
    elif cfg.concept_select_fn == "sim":
        return sim_score_select
    elif cfg.concept_select_fn == "group_mi":
        return group_mi_select
    elif cfg.concept_select_fn == "group_sim":
        return group_sim_select
    elif cfg.concept_select_fn == "diversity":
        return diversity_select
    elif cfg.concept_select_fn == "random":
        return random_select
    else:
        raise ValueError(f"Unknown concept selection function: {cfg.concept_select_fn}")

def video_asso_opt_main(cfg):
    """
    Main function for video association optimization
    
    Args:
        cfg: Configuration object
    """
    # Set the random seed
    set_seed(cfg.seed if hasattr(cfg, 'seed') else 42)
    
    # Get the concept selection function
    concept_select_fn = setup_concept_select_fn(cfg)
    
    # Load video data if enhanced initialization is used
    video_data = None
    if hasattr(cfg, 'weight_init_method') and cfg.weight_init_method in ['llm', 'continuous']:
        print(f"Loading video data for enhanced initialization with method: {cfg.weight_init_method}")
        try:
            with open(cfg.video_results_path, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
            print(f"Loaded {len(video_data)} videos from {cfg.video_results_path}")
        except Exception as e:
            print(f"Warning: Could not load video data for enhanced initialization: {e}")
            print("Falling back to default initialization")
    
    # Create the data module
    if cfg.use_dot_product:
        print("Using dot product data module (faster)")
        data_module = VideoDotProductDataModule(
            num_concept=cfg.num_concept,
            data_root=cfg.data_root,
            concept_select_fn=concept_select_fn,
            video_results_path=cfg.video_results_path,
            batch_size=cfg.batch_size,
            embedding_model=cfg.embedding_model,
            use_content_norm=cfg.use_content_norm,
            use_concept_norm=cfg.use_concept_norm,
            num_workers=cfg.num_workers,
            on_gpu=cfg.on_gpu,
            force_compute=cfg.force_compute,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            random_state=cfg.seed
        )
    else:
        print("Using standard video data module")
        data_module = VideoDataModule(
            num_concept=cfg.num_concept,
            data_root=cfg.data_root,
            concept_select_fn=concept_select_fn,
            video_results_path=cfg.video_results_path,
            batch_size=cfg.batch_size,
            embedding_model=cfg.embedding_model,
            use_content_norm=cfg.use_content_norm,
            use_concept_norm=cfg.use_concept_norm,
            num_workers=cfg.num_workers,
            on_gpu=cfg.on_gpu,
            force_compute=cfg.force_compute,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            random_state=cfg.seed
        )
    
    # Test mode
    if cfg.test:
        ckpt_path = cfg.ckpt_path
        print(f'Loading checkpoint: {ckpt_path}')
        
        # Determine model class from checkpoint or configuration
        model_class = getattr(cfg, 'model_class', 'video_asso_concept')
        print(f"Using model class: {model_class}")
        
        # Load the appropriate model
        if model_class == 'video_asso_concept_enhanced':
            if cfg.use_dot_product:
                model = VideoAssoConceptFastEnhanced.load_from_checkpoint(ckpt_path)
            else:
                model = VideoAssoConceptEnhanced.load_from_checkpoint(ckpt_path)
        else:  # Default to original model
            if cfg.use_dot_product:
                model = VideoAssoConceptFast.load_from_checkpoint(ckpt_path)
            else:
                model = VideoAssoConcept.load_from_checkpoint(ckpt_path)
        
        # Set up the trainer
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1
        )
        
        # Test the model
        trainer.test(model, data_module)
        
        # Extract the optimized concepts for each class
        threshold = cfg.threshold if hasattr(cfg, 'threshold') else 0.05
        optimized_concepts = model.extract_cls2concept(thresh=threshold)
        
        # Save the optimized concepts to a JSON file
        output_path = Path(cfg.output_dir) / 'optimized_concepts.json'
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(optimized_concepts, f, indent=4)
            
        print(f"Optimized concepts saved to {output_path}")
        
        # Log the test accuracy
        test_acc = round(float(model.total_test_acc.item()) * 100, 2)
        print(f"Test accuracy: {test_acc}%")
        
        # Save confusion matrix to CSV
        conf_matrix = model.confmat.compute().cpu().numpy()
        conf_matrix_path = Path(cfg.output_dir) / 'confusion_matrix.csv'
        np.savetxt(conf_matrix_path, conf_matrix, delimiter=',', 
                  header='predicted_not_engaging,predicted_engaging', 
                  comments='true_class,')
        print(f"Confusion matrix saved to {conf_matrix_path}")
        
        # Calculate per-class metrics
        class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        print(f"Class accuracies: not_engaging={class_acc[0]:.4f}, engaging={class_acc[1]:.4f}")
        
        return

    # Determine model class from configuration
    model_class = getattr(cfg, 'model_class', 'video_asso_concept')
    print(f"Using model class: {model_class}")
    
    # Create the model
    if model_class == 'video_asso_concept_enhanced':
        if cfg.use_dot_product:
            print("Using VideoAssoConceptFastEnhanced model with enhanced initialization")
            model = VideoAssoConceptFastEnhanced(cfg, video_data=video_data)
        else:
            print("Using VideoAssoConceptEnhanced model with enhanced initialization")
            model = VideoAssoConceptEnhanced(cfg, video_data=video_data)
    else:  # Default to original model
        if cfg.use_dot_product:
            print("Using VideoAssoConceptFast model")
            model = VideoAssoConceptFast(cfg)
        else:
            print("Using VideoAssoConcept model")
            model = VideoAssoConcept(cfg)
    
    # Set up callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.work_dir,
        filename='{epoch}-{step}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        every_n_epochs=cfg.check_val_every_n_epoch
    )
    
    # Create CSV logger instead of wandb
    csv_logger = pl.loggers.CSVLogger(
        save_dir=cfg.work_dir,
        name=f'{cfg.num_concept}concepts_{cfg.concept_select_fn}_{cfg.embedding_model}',
        flush_logs_every_n_steps=1  # Ghi log ngay lập tức
    )
    
    # Create the trainer with CSV logger
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=1,  # Log mỗi batch
        enable_progress_bar=True,  # Hiển thị progress bar
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # After training, save experiment configuration
    config_path = Path(cfg.work_dir) / 'experiment_config.json'
    with open(config_path, 'w') as f:
        # Convert config to dictionary
        config_dict = {k: v for k, v in vars(cfg).items() 
                      if not k.startswith('_') and not callable(v)}
        json.dump(config_dict, f, indent=4, default=str)
    
    print(f"Experiment configuration saved to {config_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='LaBo2 Video Classification')
    parser.add_argument('--cfg', help='Path to config file')
    parser.add_argument('--work-dir', help='Working directory to save checkpoints and config')
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--DEBUG', action='store_true', help='Debug mode')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override config options')
    
    args = parser.parse_args()
    
    if not args.test:
        cfg = pre_exp(args.cfg, args.work_dir)
    else:
        cfg = Config.fromfile(args.cfg)
        
    cfg.test = args.test
    cfg.DEBUG = args.DEBUG
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    video_asso_opt_main(cfg) 