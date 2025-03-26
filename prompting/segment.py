import torch
import librosa
from pathlib import Path
import os
import json
from scenedetect import detect, ContentDetector, AdaptiveDetector
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager


def scene_detection(video, detector):
    try:
        # Convert video path to Path object to handle spaces in filename
        video_path = Path(video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video}")

        # First try with ContentDetector
        try:
            video_manager = VideoManager([str(video_path)])
            stats_manager = StatsManager()
            scene_manager = SceneManager(stats_manager)
            
            # Add ContentDetector
            scene_manager.add_detector(ContentDetector())
            
            # Detect scenes
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            
            # If no scenes detected, try AdaptiveDetector
            if not scene_list:
                video_manager = VideoManager([str(video_path)])
                stats_manager = StatsManager()
                scene_manager = SceneManager(stats_manager)
                scene_manager.add_detector(AdaptiveDetector())
                video_manager.start()
                scene_manager.detect_scenes(frame_source=video_manager)
                scene_list = scene_manager.get_scene_list()
        
        except Exception as e:
            print(f"Advanced scene detection failed for {video}, falling back to simple detect: {str(e)}")
            # Fallback to simple detect
            scene_list = detect(str(video_path), detector)
        
        # Convert scenes to timestamps
        starts, ends = [], []
        for scene in scene_list:
            s = float(scene[0].get_timecode().split(':')[-1])
            e = float(scene[1].get_timecode().split(':')[-1])
            starts.append(s)
            ends.append(e)
        
        # If no scenes detected, use whole video duration
        if len(starts) == 0:
            starts.append(0.0)
            audio_path = video_path.parent / 'audio.wav'
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            duration = librosa.get_duration(path=str(audio_path))
            ends.append(duration)
            print(f"No scenes detected in {video}, using full duration: {duration}s")
            
        return (starts, ends)
        
    except Exception as e:
        print(f"Error in scene detection for {video}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        # Return default segmentation (whole video as one scene)
        try:
            audio_path = Path(video).parent / 'audio.wav'
            duration = librosa.get_duration(path=str(audio_path))
            print(f"Falling back to full video duration: {duration}s")
            return ([0.0], [duration])
        except Exception as e2:
            print(f"Error getting audio duration: {str(e2)}")
            print("Using default duration of 0")
            return ([0.0], [0.0])
        

def segment_with_stt_timestamp(video_dir, starts, ends):
    
    with open(video_dir + '/audio.json') as f:
        stt_result = json.load(f)
        
    new_seg = []
    mark = 0
    
    to_next = False
    # iteration over scene
    for k, (start, end) in enumerate(zip(starts, ends)):
        last_end = start
        sub_seg = {'start':[], 'end':[], 'text':[]}
        
        # segments iteration
        for m, segment in enumerate(stt_result['segments']):
            if m < mark:
                continue
            if segment['end'] < end or to_next or (segment['start'] < end and end - segment['start'] >= segment['end'] - end):
                if segment['start'] - last_end >= 1.0:
                    sub_seg['start'].append(round(last_end,2))
                    sub_seg['end'].append(round(segment['start'],2))
                    sub_seg['text'].append('')
                    last_end = segment['start']
                
                sub_seg['start'].append(round(max(segment['start'], start),2))
                sub_seg['end'].append(round(min(segment['end'], end),2))
                sub_seg['text'].append(segment['text'])
                last_end = segment['end']
                mark += 1
                to_next = False
                continue
            elif segment['start'] < end and end - segment['start'] < segment['end'] - end:
                to_next = True
            if end - last_end >= 1.0:
                sub_seg['start'].append(round(last_end,2))
                sub_seg['end'].append(round(end,2))
                sub_seg['text'].append('')
                last_end = end
            break
        
        if end - last_end >= 1.0:
            sub_seg['start'].append(round(last_end,2))
            sub_seg['end'].append(round(end,2))
            sub_seg['text'].append('')
            last_end = end
            
        if len(sub_seg['start']) >= 1:
            new_seg.append(sub_seg)
        
    return new_seg
