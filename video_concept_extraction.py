import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import openai
from typing import List, Dict
from glob import glob
import multiprocessing as mp

def normalize_path(path):
    """Normalize a path to use forward slashes."""
    return os.path.normpath(path).replace('\\', '/')

class ConceptExtractor:
    def __init__(self, args):
        # Process arguments
        self.root_dir = Path(normalize_path(args.root_dir))
        self.output_file = Path(normalize_path(args.output_json))
        self.engagement_json = args.engagement_json
        
        # Validate engagement_json parameter
        if not self.engagement_json:
            raise ValueError("--engagement_json parameter is required")
        
        # Initialize OpenAI API
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        openai.api_key = self.api_key
        self.model_name = "gpt-3.5-turbo"
        
        # Get video directories
        vid_dirs_paths = sorted(glob(os.path.join(normalize_path(str(self.root_dir)), '*')))
        self.vid_dirs = [Path(normalize_path(p)) for p in vid_dirs_paths if os.path.isdir(p)]
        self.videos = [os.path.basename(normalize_path(i)) for i in self.vid_dirs]
        
        # Load engagement data from JSON file
        self.not_engaging_ids = []
        self.engaging_ids = []
        
        if not os.path.exists(self.engagement_json):
            raise FileNotFoundError(f"Engagement JSON file not found: {self.engagement_json}")
        
        try:
            print(f"Loading engagement data from {self.engagement_json}")
            with open(self.engagement_json, 'r', encoding='utf-8') as f:
                engagement_data = json.load(f)
            
            # Get not engaging IDs from key "0"
            if "0" in engagement_data:
                self.not_engaging_ids = engagement_data["0"]
                print(f"Loaded {len(self.not_engaging_ids)} not engaging video IDs")
            else:
                print("Warning: Key '0' not found in engagement JSON file")
            
            # Get engaging IDs from key "1" for reference
            if "1" in engagement_data:
                self.engaging_ids = engagement_data["1"]
                print(f"Loaded {len(self.engaging_ids)} engaging video IDs")
            else:
                print("Warning: Key '1' not found in engagement JSON file")
                
        except Exception as e:
            raise ValueError(f"Error loading engagement data from JSON: {str(e)}")
        
        # Storage for results
        self.results = {}
        
    def _get_video_content(self, video_id):
        """Extract video content from segments.json and audio tags"""
        try:
            # Get segments.json file
            segments_file = os.path.join(normalize_path(str(self.root_dir)), 
                                        normalize_path(video_id), 'segments.json')
            
            # Extract visual and text content
            scene_descriptions = ""
            if os.path.exists(segments_file):
                with open(segments_file, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                
                for i, segment in enumerate(segments):
                    # Check if segment has valid start and end times
                    if 'start' in segment and 'end' in segment:
                        s = segment['start'][0] if isinstance(segment['start'], list) else segment['start']
                        e = segment['end'][0] if isinstance(segment['end'], list) else segment['end']
                        
                        if e - s < 0.01:
                            continue  # Skip very short scenes
                            
                    # Get visual descriptions (vcap)
                    if 'vcap' in segment and segment['vcap']:
                        vcap = segment['vcap']
                        if isinstance(vcap, list):
                            for v_item in vcap:
                                if v_item:
                                    scene_descriptions += f"Scene {i+1} - Visual: {v_item}\n"
                        else:
                            scene_descriptions += f"Scene {i+1} - Visual: {vcap}\n"
                    
                    # Get OCR text from segments
                    if 'ocr_text' in segment and segment['ocr_text']:
                        ocr_text = segment['ocr_text']
                        if isinstance(ocr_text, list):
                            if ocr_text:  # Only if the list is not empty
                                ocr_formatted = " ".join(ocr_text)
                                scene_descriptions += f"Scene {i+1} - OCR Text: {ocr_formatted}\n"
                        elif isinstance(ocr_text, str) and ocr_text:
                            scene_descriptions += f"Scene {i+1} - OCR Text: {ocr_text}\n"
                            
                    # Get speech text
                    if 'text' in segment and segment['text']:
                        text = segment['text']
                        if isinstance(text, list):
                            for t_item in text:
                                if t_item:
                                    scene_descriptions += f"Speech: {t_item}\n"
                        else:
                            scene_descriptions += f"Speech: {text}\n"
            
            # Get audio tags if available
            audio_tags = ""
            audtag_file = os.path.join(normalize_path(str(self.root_dir)), 
                                      normalize_path(video_id), 'audtag.pt')
            
            if os.path.exists(audtag_file):
                try:
                    audtag = torch.load(audtag_file)
                    audio_tags = ", ".join([tag[0] for tag in audtag[:3] if tag[1] >= 0.1])
                except:
                    pass
            
            return scene_descriptions, audio_tags
            
        except Exception as e:
            print(f"Error extracting content for video {video_id}: {str(e)}")
            return "", ""
            
    def _extract_concepts(self, video_id, video_content, audio_tags):
        """Extract concepts using GPT without revealing engagement label"""
        try:
            # Still track engagement for our records but don't tell the model
            is_engaging = video_id not in self.not_engaging_ids
            
            prompt = f"""Please analyze this video content objectively:

Video Content:
{video_content}

Audio Context: {audio_tags}

Instructions:
1. Identify 5 GENERAL concepts or characteristics present in this video based ONLY on the provided content.
2. PRIORITIZE ABSTRACT, GENERAL CONCEPTS over specific details. For example, use "educational content" instead of "physics lesson" or "storytelling" instead of "specific plot details".
3. Consider high-level patterns across all elements: visual descriptions, speech/dialog, text elements, and audio.
4. Extract concepts that represent the theme, presentation style, emotional tone, and purpose of the video.
5. AVOID overly specific details about the exact content, locations, people, or activities.
6. Use short phrases (under 10 words each phrase).
7. Your analysis must be OBJECTIVE and based solely on the provided information.

Respond with just a list of general concepts formatted like this:
- [General Concept 1]
- [General Concept 2]
- ...and so on
"""
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert video content analyzer that objectively extracts key concepts from videos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse concepts
            concepts = []
            for line in result.split("\n"):
                if line.strip().startswith("-") or line.strip().startswith("*"):
                    concept = line.strip().lstrip("- *").strip()
                    if concept:
                        concepts.append(concept)
            
            return concepts, is_engaging
            
        except Exception as e:
            print(f"Error extracting concepts: {str(e)}")
            is_engaging = video_id not in self.not_engaging_ids
            return [], is_engaging
    
    def process_video(self, video_id):
        """Process a single video to extract concepts"""
        try:
            # Get video content
            scene_desc, audio_tags = self._get_video_content(video_id)
            
            # Skip if no content found
            if not scene_desc:
                print(f"No content found for video {video_id}, skipping.")
                is_engaging = video_id not in self.not_engaging_ids
                return (video_id, {
                    'label': "engaging" if is_engaging else "not_engaging",
                    'concepts': [],
                    'content': "",
                    'audio_tags': ""
                })
            
            # Extract concepts with known engagement label
            concepts, is_engaging = self._extract_concepts(video_id, scene_desc, audio_tags)
            
            # Return results
            return (video_id, {
                'label': "engaging" if is_engaging else "not_engaging",
                'concepts': concepts,
                'content': scene_desc,
                'audio_tags': audio_tags
            })
        
        except Exception as e:
            print(f"Error processing video {video_id}: {str(e)}")
            is_engaging = video_id not in self.not_engaging_ids
            return (video_id, {
                'label': "engaging" if is_engaging else "not_engaging",
                'concepts': [],
                'content': "",
                'audio_tags': ""
            })
    
    def run(self):
        """Process all videos and extract concepts"""
        # Process in chunks to save intermediate results
        chunk_size = 10
        
        for start in range(0, len(self.videos), chunk_size):
            # Get a subset of videos to process
            video_subset = self.videos[start:start+chunk_size]
            
            # Process videos in parallel
            try:
                print(f"Processing videos {start+1}-{start+len(video_subset)} of {len(self.videos)}...")
                with mp.Pool(processes=min(mp.cpu_count(), len(video_subset))) as pool:
                    results = list(tqdm(pool.imap(self.process_video, video_subset), 
                                       total=len(video_subset)))
                
                # Update results dictionary
                self.results.update(dict(results))
                
                # Save intermediate results
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
                    
                # Small delay to prevent API rate limits
                import time
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                # Still save any results we have
                if self.results:
                    with open(self.output_file, 'w', encoding='utf-8') as f:
                        json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessing complete. Results saved to {self.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract concepts from video content with known engagement labels")
    parser.add_argument("--root_dir", type=str, default="D:/school/Thesis/ExFunTube/pipeline/videos",
                        help="Directory containing video folders with segments.json files")
    parser.add_argument("--output_json", type=str, default="video_results_3.json",
                        help="Output JSON file for concepts")
    parser.add_argument("--engagement_json", type=str, default='video_id_by_label_3.json',
                        help="Path to JSON file containing engagement data")
    
    args = parser.parse_args()
    
    extractor = ConceptExtractor(args)
    extractor.run() 