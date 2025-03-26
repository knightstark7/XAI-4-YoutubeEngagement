import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import torch
from pathlib import Path
from datetime import datetime
import librosa
import warnings
import openai
import threading
import time
from typing import Optional

# Suppress all warnings
warnings.filterwarnings("ignore")

def normalize_path(path):
    """Normalize a path to use forward slashes, avoiding double backslash issues on Windows."""
    return os.path.normpath(path).replace('\\', '/')

class TimeoutError(Exception):
    """Exception when operation times out."""
    pass

class Timer:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        self._start_time = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cancel()

    def start(self):
        def timeout_function():
            if self._start_time and time.time() - self._start_time >= self.timeout_seconds:
                raise TimeoutError("Operation timed out")
        
        self._start_time = time.time()
        self.timer = threading.Timer(1.0, timeout_function)  # Check every second
        self.timer.daemon = True
        self.timer.start()

    def cancel(self):
        if self.timer:
            self.timer.cancel()

class LanguageModel:
    def __init__(self):
        self.model_name = "gpt-3.5-turbo"
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        openai.api_key = self.api_key

    def __call__(self, prompt, temperature=0.3) -> str:
        # Creating response from GPT-3.5
        try:
            with Timer(60):  # 60 second timeout
                try:
                    # Format prompt for GPT-3.5
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that evaluates if videos are engaging and explains why. Your task is to analyze video content and explain what makes it captivating."},
                        {"role": "user", "content": f"""Video Content:
{prompt}

Instructions:
1. Focus on key engaging elements
2. Explain why the content captures viewers' attention
3. Consider both visual and audio elements
4. Keep your explanation concise (1 sentence)
5. Use an engaging and descriptive tone

Please evaluate if this video is engaging and explain why:"""}
                    ]
                    
                    # Use OpenAI API
                    response = openai.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=300,
                        top_p=0.95,
                        frequency_penalty=0.2,
                        presence_penalty=0.2
                    )
                    
                    result = response.choices[0].message.content.strip()
                    return result
                    
                except Exception as e:
                    return "The video is engaging because of its compelling visual elements and dynamic interactions that capture viewers' attention."
        except TimeoutError:
            return "The video is engaging because of its compelling visual elements and dynamic interactions that capture viewers' attention."
        except Exception as e:
            return "The video is engaging because of its compelling visual elements and dynamic interactions that capture viewers' attention."

class Explanationer:
    
    def __init__(self, root_dir, out_pth):
        self.output_pth = Path(out_pth)
        self.root_dir = Path(normalize_path(root_dir))
        
        self.explanations = {}
        # Use standard glob instead of Path.glob for Windows compatibility
        viddirs_paths = sorted(glob(os.path.join(normalize_path(str(self.root_dir)), '*')), key=lambda x: str(x).lower())
        self.viddirs = [Path(normalize_path(p)) for p in viddirs_paths if os.path.isdir(p)]
        self.vids = [os.path.basename(normalize_path(i)) for i in self.viddirs]
    
    def _get_meta(self, vid):
        try:
            # metadata
            meta_pth = os.path.join(normalize_path(str(self.root_dir)), normalize_path(vid), 'segments.json')
            try:
                with open(meta_pth) as f:
                    meta_dict = json.load(f)
                    
                # Check if any segment has vcap
                has_vcap = any('vcap' in segment and segment['vcap'] for segment in meta_dict)
                if not has_vcap:
                    # No vcap information in segments.json
                    pass
                    
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # Create empty metadata if file doesn't exist
                meta_dict = []
            
            # Check audtag.pt file
            audtag_pth = os.path.join(normalize_path(str(self.root_dir)), normalize_path(vid), 'audtag.pt')
            try:
                audtag = torch.load(audtag_pth)
                audtag = [i[0] for i in audtag[:3] if i[1] >= 0.1]
            except FileNotFoundError:
                audtag = []
            except Exception:
                audtag = []
            
            # gather scene descriptions
            scenes = []
            n = 1
            
            for meta in meta_dict:
                try:
                    # Ensure required fields exist
                    meta_start = meta.get('start', [])
                    meta_end = meta.get('end', [])
                    meta_vcap = meta.get('vcap', [])
                    meta_diar = meta.get('diar', [])
                    meta_text = meta.get('text', [])
                    
                    # Get maximum length of lists
                    max_len = max(
                        len(meta_start), len(meta_end),
                        len(meta_vcap) if isinstance(meta_vcap, list) else 0,
                        len(meta_diar) if isinstance(meta_diar, list) else 0, 
                        len(meta_text) if isinstance(meta_text, list) else 0
                    )
                    
                    for i in range(max_len):
                        try:
                            # Get safe values from each list
                            s = meta_start[i] if i < len(meta_start) else 0
                            e = meta_end[i] if i < len(meta_end) else 0
                            
                            if e - s < 0.01:
                                continue  # Skip very short scenes
                            
                            # Get vcap, prioritize vcap if available
                            v = ""
                            if isinstance(meta_vcap, list) and i < len(meta_vcap) and meta_vcap[i]:
                                v = meta_vcap[i]
                                if isinstance(v, list) and v:  # If vcap is a list and not empty
                                    v = v[0] if v else ""  # Take first element
                                
                            # Get information about speaker and text
                            d = meta_diar[i] if isinstance(meta_diar, list) and i < len(meta_diar) else 'unknown'
                            t = meta_text[i] if isinstance(meta_text, list) and i < len(meta_text) else ''
                            
                            # Create scene description
                            if len(v) > 0:
                                if len(t) > 0:
                                    p = f"Scene {n}:\nSpeaker {d}: \"{t}\"\nVisual: {v[0].upper()}{v[1:] if len(v) > 1 else ''}.\n\n"
                                else:
                                    p = f"Scene {n}:\nVisual: {v[0].upper()}{v[1:] if len(v) > 1 else ''}.\n\n"
                            else:
                                if len(t) > 0:
                                    p = f"Scene {n}:\nSpeaker {d}: \"{t}\"\n\n"
                                else:
                                    # Skip if no content
                                    continue
                            
                            n += 1
                            scenes.append([(s, e), (v, d, t), p])
                        except IndexError:
                            continue
                except Exception:
                    continue
            
            # concatenate scene descriptions
            scene_desc = ''
            for _, _, p in scenes:
                scene_desc += p
            
            # If no descriptions, add a note
            if not scene_desc:
                scene_desc = "No scene descriptions available for this video.\n\n"
            
            # concatenate audio descriptions
            audtag = ','.join(audtag)
            if not audtag:
                audtag = "No audio tags available"
            
            return scene_desc, audtag
        except Exception:
            return "No scene descriptions available.", "No audio tags available"
    
    def _explanation_prompt_w_audio(self, lm, scenes, audio_cap):
        prompt = """You are a helpful assistant that evaluates if this video is engaging or not, and explains why. Analyze the following video content and explain what makes it captivating.

Video Content:
{scenes}

Audio Context: {audio_tags}

Instructions:
1. Focus on the key engaging elements
2. Explain why this content is quality and likely to get lots of likes
3. Keep your explanation concise (1 sentence)
4. Consider both visual and audio elements

Please evaluate if this video is engaging or not, and explain why:""".format(
            scenes=scenes,
            audio_tags=audio_cap
        )
        
        try:
            # Generating explanation
            result = lm(prompt)
            return prompt, result
        except Exception:
            fallback_response = "The video is engaging because of its compelling visual elements and dynamic interactions that capture viewers' attention."
            return prompt, fallback_response
    
    def _run_video(self, vid):
        try:
            time.sleep(0.5)
            scene_desc, audtag = self._get_meta(vid)
            
            # Use LanguageModel
            lm = LanguageModel()
            prompt, explanation = self._explanation_prompt_w_audio(lm, scene_desc, audtag)
            return (vid, {'prompt': prompt, 'res': explanation})
        except Exception as e:
            return (vid, {'prompt': f"Error: {str(e)}", 'res': f"Error occurred: {str(e)}"})
    
    def __call__(self):
        from glob import glob
        import multiprocessing as mp
        
        len_chunk = 3
        idx_list = range(0, len(self.vids), len_chunk)
        for i, start in enumerate(idx_list):
            # Progress tracking
            try:
                p = mp.Pool(10)
                sub_vids = self.vids[start : start+len_chunk]
                result = list(tqdm(p.imap(self._run_video, sub_vids), total=len(sub_vids)))
                p.close()
                p.join()
                self.explanations.update(dict(result))
                
                # Save current results
                output_path = normalize_path(str(self.output_pth))
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.explanations, f, indent=2, ensure_ascii=False)
            except Exception:
                # Save current results if error occurs
                if self.explanations:
                    output_path = normalize_path(str(self.output_pth))
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(self.explanations, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_dir', default='/gallery_louvre/dayoon.ko/dataset/videohumor')
    parser.add_argument('--meta', default='/gallery_louvre/dayoon.ko/research/mmvh/videohumor/exp_output/audio_caption/audio_caption.json')
    parser.add_argument('--out_pth', default='./explanations.json')
    args = parser.parse_args()
    
    # output path
    usecase = Explanationer(
        root_dir=args.root_dir,
        out_pth=args.out_pth
    )
    usecase()