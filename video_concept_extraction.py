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
        
        # Initialize OpenAI API
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        openai.api_key = self.api_key
        self.model_name = "gpt-4o"
        
        # Get video directories
        vid_dirs_paths = sorted(glob(os.path.join(normalize_path(str(self.root_dir)), '*')))
        self.vid_dirs = [Path(normalize_path(p)) for p in vid_dirs_paths if os.path.isdir(p)]
        self.videos = [os.path.basename(normalize_path(i)) for i in self.vid_dirs]
        
        # List of not engaging video IDs
        self.not_engaging_ids = [
            "0c38722311df0412921f683d18354044", 
            "14cef79fee6e5d55c12c855025c0d144", 
            "21cdfb16bc352b1dae32da0c1fa313dc", 
            "375be1c7fe6a88682bfafa55c083de6e", 
            "73820e5bd5823024e2511184dd619db4", 
            "b2350338e5687ed11c7fa90dab03919a", 
            "ce820a1e9b9e68997d190cd1dc61c098", 
            "d15be6cd1c8e31623fedff9da7458b2a", 
            "e7eb609bc9c30387d6b5e4205fce3c18", 
            "e47da965aee26c70fdf0424e92298985", 
            "e8562abeb89351f93fcc55e155aff8a6",          
            "3dd1eb65f3b4aef0a8ab9a59fc109dfd",
            "ce9e4ea25e5ae6a0265fa2ed98da36ae",
            "88d98e814f885730dfb1e12cceba0e73",
            "e450f3cb167a794c095ffc56aa0e5ace",
            "cb4e73850c5fe6ebdbc89bb877761b95",
            "7f8cb092dd8a3606b23374019bf0cabb",
            "d8d836fe8761bfa207f2a9923e4ee970",
            "eb042efe930e4fbece0354d5d8618569",
            "f832f6fd58e0e9dbbfe3ff8103875c1f",
            "a0897c7b1a3c321e1c8c70af44d3090a",
            "11eb90fcf0bfd1494b4ff2b19b6514be",
            "e45aaa56bcc19b9ccf33c530a0eecaa0",
            "404eaf447d7dd6221498ccf972c2cfa0",
            "f17f972d9fe4512471ae21dcb7d2594c",
            "0d2019cbdb9c2f1bc6ef005fd0a23229",
            "b2aae9e7f35848c22039d0d57a9d40c2",
            "a2f180b138257cc9dc424d367ea6723d",
            "ef26fd368ceebce3ffda1e0c02a51a15",
            "e571b688efae2bdd4663129429023640",
            "a632418aaa7bacb5970d9f8b9db5ad1d",
            "2f70c3f7f70901787b728574e467d869",
            "e625df03cd453344ac7224842180d1be",
            "e7e1b9d39d25fc8519960bf59cdb5782",
            "cbae542613bc37c1f2c46f754e2b9de2",
            "eb81b1215afedbf8ad5c13f178cb9ddb",
            "b9d4a301eecbc3aac058fcf11de7be51",
            "eefc85bd90c4bb8b6d6c619e3ea855d2",
            "bbfbdbb61347bedbaf24a1f6e6392928",
            "86ccc905e1f696bd24247d451217c85a",
            "11037ab7e13410b6c0385a94e48f6bf5"
        ]
        
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
        """Extract concepts using GPT with known engagement label"""
        try:
            # Determine if this video is engaging
            is_engaging = video_id not in self.not_engaging_ids
            engagement_label = "engaging" if is_engaging else "NOT engaging"
            
            prompt = f"""Please analyze this video content:

Video Content:
{video_content}

Audio Context: {audio_tags}

This video is labeled as {engagement_label}.

Instructions:
1. Based on the given label ({engagement_label}), identify 5-10 concepts or characteristics present in this video that are typical for {engagement_label} videos.
2. Focus on aspects that contribute to the video being {engagement_label}.
3. Pay attention to all elements: visual descriptions, OCR text (on-screen text), speech/dialog, and audio context.
4. Consider more general about video content, ignore specific detail related to visual and audio quality.
5. Use short phrases (under 10 words each phrase).

Respond with just a list of concepts formatted like this:
- [Concept 1]
- [Concept 2]
- ...and so on
"""
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert video content analyzer that extracts concepts based on engagement labels."},
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
        chunk_size = 5
        
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
    parser.add_argument("--output_json", type=str, default="video_results.json",
                        help="Output JSON file for concepts")
    
    args = parser.parse_args()
    
    extractor = ConceptExtractor(args)
    extractor.run() 