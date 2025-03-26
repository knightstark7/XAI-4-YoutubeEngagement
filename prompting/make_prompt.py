import argparse
from glob import glob
from pathlib import Path
from segment import scene_detection, segment_with_stt_timestamp
from run_blip import load_model, caption_video
import sys
import os
import gc
import torch

def normalize_path(path):
    """Normalize a path to use forward slashes, avoiding double backslash issues on Windows."""
    return os.path.normpath(path).replace('\\', '/')

sys.path.append('InternVideo/Downstream/Video-Text-Retrieval')
from inference import caption_retrieval
sys.path.append('EfficientAT')
from infer_custom import audio_tagging
from scenedetect import AdaptiveDetector
from diarization import run_diarization
from tqdm import tqdm
import json

class Prompter:
    
    def __init__(self, args):
        self.device = args.device
        self.adaptive_threshold = args.adaptive_threshold #1.5
        self.min_scene_len = args.min_scene_len #15
        self.window_width = args.window_width #4
        self.min_content_val = args.min_content_val #6
        self.root_dir = normalize_path(args.root_dir)  # Normalize the root directory path
        
        # Get all video directory paths
        all_video_dirs = [Path(normalize_path(i)) for i in sorted(glob(args.root_dir + '/*')) if os.path.isdir(i)]
        
        # Limit the number of videos to process if limit_videos parameter is provided
        self.limit_videos = getattr(args, 'limit_videos', None)
        if self.limit_videos and self.limit_videos > 0 and self.limit_videos < len(all_video_dirs):
            self.video_dirs = all_video_dirs[:self.limit_videos]
        else:
            self.video_dirs = all_video_dirs
            
        # Find video files in directories
        self.videos = []
        for vdir in self.video_dirs:
            mp4_files = glob(str(vdir / '*.mp4'))
            if mp4_files:
                self.videos.append(mp4_files[0])
        
        self.caption_filename = 'caption_corpus.pt'
        self.segmentation_filename = 'segments.json'
        self.audcap_filename = 'audtag.pt'
        
    def _clean_memory(self):
        """Clean up memory after each processing step"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    #####################
    # 1. Segment videos #
    #####################
    def _segment_videos(self):
        Detector = AdaptiveDetector(adaptive_threshold=self.adaptive_threshold,
                                    min_scene_len=self.min_scene_len,
                                    window_width=self.window_width,
                                    min_content_val=self.min_content_val)
        for video_dir, video in zip(self.video_dirs, self.videos):
            try:
                result = scene_detection(str(video), Detector)
                if result:
                    starts, ends = result
                    result = segment_with_stt_timestamp(str(video_dir), starts, ends)
                    with open(video_dir / self.segmentation_filename, 'w') as f:
                        json.dump(result, f, indent=2)
            except Exception as e:
                continue
        
        self._clean_memory()
    
    #########################
    # 2. Visual Description #
    #########################
    def _generate_caption_corpus(self):
        try:
            # First try to load the model on the specified device
            initial_device = self.device
            model, vis_processors, actual_device = load_model(initial_device)
            
            for video_dir, video in tqdm(zip(self.video_dirs, self.videos)):
                outpth = str(video_dir / self.caption_filename)
                if os.path.exists(outpth):
                    continue
                try:
                    # Process each video, note that the device may change during this process
                    captions = caption_video(video, model, vis_processors, actual_device)
                    torch.save(captions, outpth)
                    
                    # Clean up memory after each video
                    self._clean_memory()
                except Exception:
                    # Continue with next video
                    pass
        except Exception:
            # Some videos may not have caption data
            pass
            
        # Clean up memory after completion
        self._clean_memory()
    
    def _select_frame_captions(self):
        try:
            # Normalize paths
            root_dir = normalize_path(str(self.root_dir))
            caption_filename = normalize_path(self.caption_filename)
            segmentation_filename = normalize_path(self.segmentation_filename)
            
            # Handle case of duplicate 'videos' in path
            if 'videos/videos' in root_dir or 'videos\\videos' in root_dir:
                root_dir = root_dir.replace('videos/videos', 'videos').replace('videos\\videos', 'videos')
            
            # Check if root directory exists
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"Root directory not found: {root_dir}")
            
            # Check if there are any video directories
            video_dirs = [d for d in glob(os.path.join(root_dir, '*')) if os.path.isdir(d)]
            if not video_dirs:
                raise FileNotFoundError(f"No video directories found in: {root_dir}")
            
            # Call caption_retrieval with normalized path
            caption_retrieval(root_dir, caption_filename, segmentation_filename)
        except Exception:
            pass
        finally:
            self._clean_memory()

    #############
    # 3. Speech #
    #############
    def _speaker_diarization(self):
        try:
            run_diarization(self.root_dir, self.segmentation_filename)
        except Exception:
            pass
        finally:
            self._clean_memory()
    
    ############
    # 4. Audio #
    ############
    def _audio_tagging(self):
        try:
            # Check if audio.wav files exist for the videos
            audio_files = glob(f"{normalize_path(self.root_dir)}/*/audio.wav")
            video_dirs = glob(f"{normalize_path(self.root_dir)}/*")
            video_dirs = [d for d in video_dirs if os.path.isdir(d)]
            
            if len(audio_files) == 0:
                # Create dummy audtag.pt file for each directory
                for vdir in video_dirs:
                    try:
                        empty_result = []
                        save_path = os.path.join(vdir, 'audtag.pt')
                        torch.save(empty_result, save_path)
                    except Exception:
                        pass
            else:
                # If audio files exist, call audio_tagging normally
                audio_tagging(normalize_path(self.root_dir), self.audcap_filename)
        except Exception:
            # Create dummy audtag.pt file
            try:
                video_dirs = glob(f"{normalize_path(self.root_dir)}/*")
                video_dirs = [d for d in video_dirs if os.path.isdir(d)]
                for vdir in video_dirs:
                    try:
                        empty_result = []
                        save_path = os.path.join(vdir, 'audtag.pt')
                        if not os.path.exists(save_path):
                            torch.save(empty_result, save_path)
                    except Exception:
                        pass
            except Exception:
                pass
        finally:
            self._clean_memory()
        
    def __call__(self):
        # Clean memory
        self._clean_memory()
        
        try:
            self._segment_videos()
            self._speaker_diarization()
            self._generate_caption_corpus()
            self._select_frame_captions()
            self._audio_tagging()
        except Exception:
            pass
        finally:
            self._clean_memory()
        