import argparse
from make_prompt import Prompter
from explanation import Explanationer
import os
import gc
import torch
import sys
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: all, 1: info, 2: warnings, 3: errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Disable huggingface/transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='D:/school/Thesis/ExFunTube/pipeline/videos')
    parser.add_argument('--device', default='cuda')  # Default to use GPU
    parser.add_argument('--adaptive_threshold', type=float, default=1.5)
    parser.add_argument('--min_scene_len', type=int, default=15)
    parser.add_argument('--window_width', type=int, default=4)
    parser.add_argument('--min_content_val', type=int, default=6)
    parser.add_argument('--out_pth', default='explanation.json')
    parser.add_argument('--limit_videos', type=int, default=None, help='Limit the number of videos to process')
    parser.add_argument('--use_cpu', action='store_true', help='Force using CPU instead of CUDA', default=False)
    parser.add_argument('--skip_explanation', action='store_true', help='Skip the explanation step')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU for processing (default is True)')
    parser.add_argument('--skip_segment', action='store_true', help='Skip the segment creation step')
    parser.add_argument('--skip_vcap', action='store_true', help='Skip the vcap creation step')
    args = parser.parse_args()
    
    # Normalize root_dir path for Windows
    args.root_dir = os.path.normpath(args.root_dir).replace('\\', '/')
    
    # Check and manage memory
    if args.use_gpu:
        args.use_cpu = False  # If requested to use GPU, turn off use_cpu flag
        
    if not args.use_cpu:
        # Check if CUDA is available
        if torch.cuda.is_available():
            # Check CUDA memory
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                free_memory_gb = free_memory / (1024 ** 3)  # Convert to GB
                
                # Free CUDA memory
                torch.cuda.empty_cache()
                args.device = 'cuda'
            except Exception as e:
                args.device = 'cuda'  # Still use GPU if there's an error
        else:
            args.device = 'cpu'
    else:
        args.device = 'cpu'
    
    return args

if __name__ == "__main__":
    args = get_args()
    
    # 1. First create segments using prompter (if not skipped)
    if not args.skip_segment:
        try:
            prompter = Prompter(args)
            prompter()
        except Exception as e:
            pass
    
    # Free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Create vcap using InternVideo (only if not skipping this step)
    if not args.skip_explanation and not args.skip_vcap:
        try:
            # Make sure current directory is in path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)

            # Import caption_retrieval from InternVideo
            from InternVideo.Downstream.Video_Text_Retrieval.inference import caption_retrieval
            
            # Paths and parameters
            caption_filename = "caption_corpus.pt"
            segmentation_filename = "segments.json"
            
            # Call vcap creation function
            caption_retrieval(args.root_dir, caption_filename, segmentation_filename)
        except ImportError:
            pass
        except Exception as e:
            pass
    
    # Free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3. Create explanations from segment + vcap (if not skipped)
    if not args.skip_explanation:
        try:
            # Default use GPU unless --use_cpu parameter is provided
            force_cpu = args.use_cpu
            explanationer = Explanationer(args.root_dir, args.out_pth)
            explanationer()
        except Exception as e:
            pass