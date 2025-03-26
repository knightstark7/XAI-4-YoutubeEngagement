import json
import openai
import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import torch
import signal
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from glob import glob
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Diarization:
    def __init__(self, lm, meta_path):
        self.lm = lm
        self.metas = [i for i in sorted(glob(str(Path(meta_path))), key=lambda x:x.lower()) if os.path.exists(i)]
        self.vids = [str(Path(i).parent.name) for i in self.metas]
        self.js = {}
        
    def _get_transcript(self, meta):
        transcript = ''
        transcript_list = []
        for i, scene in enumerate(meta):
            for k, utter in enumerate(scene['text']):
                if len(utter) > 0:
                    transcript += f'{utter}' + '\n'
                    transcript_list.append(utter)
        transcript = 'Transcript: ' + transcript
        return transcript, transcript_list, len(transcript_list)
    

    def _make_num_speaker_prompt(self, transcript):
        # Improve prompt for better speaker count detection
        prompt1 = [
            {"role": "system", "content": "You are a helpful assistant specialized in analyzing conversations and identifying distinct speakers."},
            {"role": "user", "content": f"""Please analyze this transcript and determine the number of unique speakers.
Consider these points:
- Look for different speaking styles and patterns
- Consider context and conversation flow
- Count unique speakers even if they speak only once
- Ignore narrator or system messages

Transcript:
{transcript}

How many unique speakers are in this transcript? Return ONLY a single number."""}
        ]
        
        # Try multiple times with different approaches if needed
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                res1 = self._run_lm(prompt1)
                # Extract number from response
                if res1:
                    numbers = re.findall(r'\d+', res1)
                    if numbers:
                        # Get the first number found
                        speaker_count = int(numbers[0])
                        # Ensure at least 1 speaker
                        if speaker_count < 1:
                            speaker_count = 1
                        return prompt1, str(speaker_count)
                    
                # If no number found, modify prompt for next attempt
                prompt1[1]["content"] += "\nPlease respond with ONLY a number."
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                continue
                
        # Default to 2 speakers if all attempts fail
        print("Failed to detect speaker count, defaulting to 2")
        return prompt1, "2"

    def _make_diarization_prompt(self, prompt1, res1):
        prompt2 = f'Based on the context of the given transcript and your estimated number of speakers, please assign speakers to the transcript.'
        prompt2 = [
                    {"role" : "system", "content" : "You are a helpful assistant."},
                    {"role" : "user", "content" : prompt1[-1]['content']},
                    {"role" : "user", "content" : res1},
                    {"role" : "user", "content" : prompt2}
                ]
        res2 = self._run_lm(prompt2)
        return prompt2, res2
    

    def _make_diarization_prompt_sequentially(self, transcript_list, prompt1, res1):
        try:
            if not transcript_list or len(transcript_list) == 0:
                print("Warning: Empty transcript list")
                return [], []
                
            output = ['Speaker 1']
            if len(transcript_list) > 1:
                # Ensure transcript items are strings
                first_sentence = str(transcript_list[0]) if transcript_list[0] else ""
                second_sentence = str(transcript_list[1]) if transcript_list[1] else ""
                
                prompt2 = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt1},
                    {"role": "assistant", "content": res1},
                    {"role": "user", "content": f"Please let me know the most likely speaker number given a sentence in the transcript. "\
                                              f"Example) Sentence: {first_sentence} Speaker Number : Speaker 1. Sentence: {second_sentence}"}
                ]
                
                try:
                    res2 = self._run_lm(prompt2)
                    if res2:
                        output.append(res2)
                        prompt2.append({"role": "assistant", "content": res2})
                except Exception as e:
                    print(f"Error processing initial dialogue: {str(e)}")
                    return prompt2, output
            
            # Process remaining sentences
            for s in transcript_list[2:]:
                try:
                    sentence = str(s) if s else ""
                    prompt2.append({"role": "user", "content": f'Sentence: {sentence}'})
                    res2 = self._run_lm(prompt2)
                    if res2:
                        output.append(res2)
                        prompt2.append({"role": "assistant", "content": res2})
                except Exception as e:
                    print(f"Error processing sentence: {str(e)}")
                    continue
                    
            return prompt2, output
            
        except Exception as e:
            print(f"Error in make_diarization_prompt_sequentially: {str(e)}")
            return [], []

    def _run_lm(self, prompt):
        try:
            # Convert prompt to a format suitable for T5
            if isinstance(prompt, list):
                # Convert chat format to single string
                conversation = []
                for msg in prompt:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if content:
                        if role == "system":
                            conversation.append(f"System: {content}")
                        elif role == "user":
                            conversation.append(f"User: {content}")
                        elif role == "assistant":
                            conversation.append(f"Assistant: {content}")
                prompt_text = "\n".join(conversation)
            else:
                prompt_text = str(prompt)
                
            return self.lm(prompt_text)
            
        except Exception as e:
            print(f"Error in _run_lm: {str(e)}")
            return None

    def _is_not_dialogue(self, res):
        if 'one' in res or '1' in res or '0' in res:
            return True
        return False

    def find_number(self, string):
        for word in string.split():
            if word.isnumeric():
                return int(word)


    def get_speaker(self, text):
        """Extract speaker information from text"""
        try:
            # Initialize result
            speakers = []
            current_speaker = []
            
            # Split text into lines and process
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for speaker indicators
                speaker_match = re.search(r'Speaker\s*(\d+)|Person\s*(\d+)|Voice\s*(\d+)', line, re.IGNORECASE)
                if speaker_match:
                    if current_speaker:
                        speakers.append(' '.join(current_speaker))
                        current_speaker = []
                    continue
                    
                current_speaker.append(line)
                
            # Add last speaker's lines
            if current_speaker:
                speakers.append(' '.join(current_speaker))
                
            return speakers if speakers else [""]
            
        except Exception as e:
            print(f"Error in get_speaker: {str(e)}")
            return [""]

    def get_speaker_seq(self, text):
        """Extract speaker number from sequential analysis"""
        try:
            # Look for speaker number in various formats
            patterns = [
                r'Speaker\s*(\d+)',
                r'Person\s*(\d+)',
                r'Voice\s*(\d+)',
                r'#(\d+)',
                r'(\d+):'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    number = int(match.group(1))
                    return number if number > 0 else 1
                    
            # If no speaker number found, try to find any number
            numbers = re.findall(r'\d+', text)
            if numbers:
                number = int(numbers[0])
                return number if number > 0 else 1
                
            return 1  # Default to speaker 1
            
        except Exception as e:
            print(f"Error in get_speaker_seq: {str(e)}")
            return 1

    
    def _post_processing(self, meta, num, gpt_result=None, sequential=False):
        # Check if meta is None
        if meta is None:
            print("Warning: meta is None in _post_processing")
            return None
            
        try:
            # monologue
            if num == 1:
                for i, scene in enumerate(meta):
                    diar = []
                    for utter in scene['text']:
                        if len(utter) == 0:
                            diar.append("")
                        else:
                            diar.append("1")
                    meta[i]['diar'] = diar
                return meta
            
            # dialogue
            elif not sequential:
                # split result and assign speakers
                if gpt_result is None:
                    print("Warning: gpt_result is None for dialogue processing")
                    return None
                    
                speakers = self.get_speaker(gpt_result)
                
                # check whether the result is okay
                total = 0
                for s in speakers:
                    total += len(s)
                if total < num:
                    print(f"Warning: total speakers ({total}) less than expected ({num})")
                    return None
                
                # if ok, update meta data
                for i, scene in enumerate(meta):
                    diar = []
                    for utter in scene['text']:
                        if len(utter) == 0:
                            diar.append('')
                            continue
                        speaker_found = False
                        for speaker, string in enumerate(speakers):
                            if utter.lower().strip() in string.lower():
                                diar.append(f'{speaker + 1}')
                                speaker_found = True
                                break
                        if not speaker_found:
                            diar.append('1')  # Default to speaker 1 if no match found
                    meta[i]['diar'] = diar
                return meta
                
            # dialogue with sequential result
            else:
                if gpt_result is None:
                    print("Warning: gpt_result is None for sequential dialogue processing")
                    return None
                    
                res = []
                b_speaker = 1
                for g_res in gpt_result:
                    if g_res is None:
                        print("Warning: Found None result in gpt_result")
                        res.append(str(b_speaker))
                        continue
                        
                    speaker = self.get_speaker_seq(g_res)
                    if speaker > 0:
                        res.append(str(speaker))
                        b_speaker = speaker
                    else:
                        res.append(str(b_speaker))

                mark = 0
                for i, scene in enumerate(meta):
                    diar = []
                    for utter in scene['text']:
                        if len(utter) > 0:
                            if mark < len(res):
                                diar.append(str(res[mark]))
                                mark += 1
                            else:
                                print(f"Warning: mark ({mark}) exceeds res length ({len(res)})")
                                diar.append(str(b_speaker))
                        else:
                            diar.append('')
                    meta[i]['diar'] = diar
                return meta
                
        except Exception as e:
            print(f"Error in _post_processing: {str(e)}")
            return None
        
    
    def __call__(self):
        
        for i, aud in tqdm(enumerate(self.metas), total= len(self.metas)):
            
            with open(aud) as f:
                meta = json.load(f)
            
            # Keep a copy of original meta
            original_meta = meta.copy()
            
            transcript, transcript_list, num = self._get_transcript(meta)
            
            # if no transcript
            if num == 0: 
                continue
            
            # if only one utterance
            if num == 1:
                meta = self._post_processing(meta, 1)
                
            # if there are more than two utterances
            else:
                prompt1, res1 = self._make_num_speaker_prompt(transcript)
                
                # Force at least 2 speakers for better results
                num_speakers = int(res1) if res1.isdigit() else 2
                if num_speakers < 2:
                    num_speakers = 2
                
                # Assign speakers
                prompt2, res2 = self._make_diarization_prompt(prompt1, str(num_speakers))
                meta = self._post_processing(meta, num_speakers, res2)
                
                # If processing failed, try sequential approach
                if meta is None:
                    prompt3, res3 = self._make_diarization_prompt_sequentially(transcript_list, prompt1, str(num_speakers))
                    meta = self._post_processing(original_meta.copy(), num_speakers, res3, sequential=True)
                
                # If all methods failed, default to assigning all text to Speaker 1
                if meta is None:
                    print(f"All diarization methods failed for {aud}, defaulting to single speaker")
                    meta = self._post_processing(original_meta.copy(), 1)

            # Ensure meta is saved even if all methods failed
            if meta is None:
                print(f"Using original meta with default speaker assignment for {aud}")
                meta = original_meta
                # Set all utterances to Speaker 1
                for i, scene in enumerate(meta):
                    diar = []
                    for utter in scene['text']:
                        diar.append("1" if len(utter) > 0 else "")
                    meta[i]['diar'] = diar
            
            # Save meta to file
            with open(aud, 'w') as f:
                json.dump(meta, f, indent=2)
    
    
class LanguageModel:
    def __init__(self, key=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load Flan-T5 model - better for instruction following
        model_name = "google/flan-t5-large"  # Options: base, small, large, xl, xxl
        print(f"Loading {model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Load model with lower precision to save memory if on GPU
            if self.device.type == 'cuda':
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to smaller model if necessary
            model_name = "google/flan-t5-base"
            print(f"Falling back to {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def __call__(self, prompt, engine=None, temperature=0.1) -> str:
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Format prompt appropriately for Flan-T5
                if isinstance(prompt, list):
                    # For counting speakers
                    if any("analyze this transcript" in msg.get('content', '').lower() for msg in prompt):
                        input_text = "Count the number of different speakers in this transcript: " + \
                                    next((msg['content'] for msg in prompt if "transcript" in msg.get('content', '').lower()), "")
                    
                    # For sequential speaker detection    
                    elif any("most likely speaker number" in msg.get('content', '').lower() for msg in prompt):
                        sentence = prompt[-1]['content'].replace("Sentence: ", "")
                        input_text = f"Identify the speaker number for this sentence: {sentence}"
                    
                    # For assigning speakers
                    elif any("assign speakers" in msg.get('content', '').lower() for msg in prompt):
                        transcript = next((msg['content'] for msg in prompt if "transcript" in msg.get('content', '').lower()), "")
                        num_speakers = next((msg['content'] for msg in prompt if msg.get('role') == 'user' and re.search(r'\d+', msg['content'])), "2")
                        input_text = f"Assign {num_speakers} speakers to this transcript: {transcript}"
                    
                    else:
                        # Combine all content
                        input_text = " ".join([msg.get('content', '') for msg in prompt])
                else:
                    input_text = prompt

                # Add explicit instruction for speaker tasks
                if "count" in input_text.lower() and "speaker" in input_text.lower():
                    input_text += " Return only the number."
                    
                # Tokenize with appropriate settings
                inputs = self.tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    max_length=1024,
                    truncation=True
                ).to(self.device)
                
                # Generate with appropriate settings for the task
                if "count" in input_text.lower() and "speaker" in input_text.lower():
                    # More focused generation for counting
                    outputs = self.model.generate(
                        **inputs,
                        max_length=10,
                        min_length=1,
                        num_beams=4,
                        temperature=0.7,
                        do_sample=False
                    )
                else:
                    # More creative generation for dialog
                    outputs = self.model.generate(
                        **inputs,
                        max_length=512,
                        min_length=10,
                        num_beams=5,
                        temperature=temperature,
                        top_p=0.95,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=2
                    )
                
                # Decode the generated text
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Post-process for specific tasks
                if "count" in input_text.lower() and "speaker" in input_text.lower():
                    # Extract number from response
                    numbers = re.findall(r'\d+', response)
                    if numbers:
                        # Default to a reasonable number if unreasonable
                        speaker_count = int(numbers[0])
                        if speaker_count > 10:  # Unlikely to have >10 speakers in short transcript
                            return "2"
                        elif speaker_count < 1:
                            return "1" 
                        return numbers[0]
                    return "2"  # Default speakers
                
                return response
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"CUDA out of memory, clearing cache and retrying: {e}")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    retry_count += 1
                    continue
                else:
                    print(f"Runtime error: {e}")
                    retry_count += 1
                    time.sleep(1)
                    continue
                    
            except Exception as e:
                print(f"Attempt {retry_count + 1} failed: {e}")
                retry_count += 1
                time.sleep(1)
                continue
                
        print("Failed after maximum retries, returning default response")
        # Return safe defaults
        if "count" in str(prompt).lower() and "speaker" in str(prompt).lower():
            return "2"
        return "Speaker 1"


def run_diarization(root_dir, segmentation_filename):  
    api_key = os.getenv("OPENAI_API_KEY")
    gpt3_api = LanguageModel(api_key)
    meta_path = root_dir + '/*/' + segmentation_filename
    usecase = Diarization(
        lm=gpt3_api,
        meta_path=meta_path,
    )
    usecase()
        
