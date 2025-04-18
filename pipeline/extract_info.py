'''
    Extract audio, and perform speech-to-text and video captioning.
    Finally, gather all the results and store them as a json file.
'''

import argparse
import os
from glob import glob
import json
import torch
from multiprocessing import Pool
from pathlib import Path
from faster_whisper import WhisperModel
from zero_shot_video_to_text.run import run_videos
from tqdm import tqdm


########## Extract Audios ##########
def extract_audio(vid_dir):
    # extract audio from given video dir
    try:
        aud_dir = Path(vid_dir).parent / "audio.wav"
        if not os.path.exists(aud_dir):
            print(f"Extracting audio from {vid_dir}")
            os.system(f'ffmpeg -y -i "{vid_dir}" {aud_dir}')
        else:
            print(f"Audio already exists for {vid_dir}, skipping")
    except Exception as e:
        print(f'Error extracting audio for {vid_dir}: {str(e)}')

def extract_audios(args):
    # args.root_dir
    # |- {id}
    #    |- {title}.mp4
    #    |-  audio.wav
    videos = glob(os.path.join(args.root_dir, '*/*.mp4'))
    if len(videos) == 0:
        print('Warning: No videos found in directory.')
        return
    
    print(f"Found {len(videos)} videos for audio extraction")
    
    with Pool(args.num_workers) as p:
        p.map(extract_audio, videos)


########## Speech-to-Text ##########
def run_whisper(args):
    # Tìm tất cả thư mục video
    video_dirs = [i for i in glob(os.path.join(args.root_dir, '*')) if os.path.isdir(i)]
    
    if len(video_dirs) == 0:
        print('Warning: No video directories found.')
        return
    
    print(f"Found {len(video_dirs)} video directories for speech-to-text processing")
    
    # Chọn device và kiểu tính toán (ví dụ: float16 cho GPU Nvidia)
    # compute_type có thể là: float16, int8_float16, int8, float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    model_size = "large-v2"
    
    # Tải mô hình Faster-Whisper
    # Download model tự động vào cache nếu chưa có
    print(f"Loading Faster-Whisper model: {model_size} on {device} with {compute_type}")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Faster-Whisper model: {e}")
        print("Please ensure CUDA/cuDNN are installed correctly if using GPU.")
        print("Falling back to CPU...")
        try:
            device = "cpu"
            compute_type = "int8" # Dùng int8 trên CPU để nhanh hơn
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("Model loaded successfully on CPU.")
        except Exception as cpu_e:
            print(f"Failed to load model even on CPU: {cpu_e}")
            return # Thoát nếu không tải được model
    
    for vid_dir in tqdm(video_dirs, desc="Processing speech-to-text"):
        audio_file = os.path.join(vid_dir, 'audio.wav')
        json_file = os.path.join(vid_dir, 'audio.json')
        
        # Skip if JSON already exists or audio doesn't exist
        if os.path.exists(json_file):
            # print(f"Speech-to-text already exists for {vid_dir}, skipping") # Giảm log thừa
            continue
        
        if not os.path.exists(audio_file):
            # print(f"Audio file not found for {vid_dir}, skipping") # Giảm log thừa
            continue
        
        # transcribe audio and store result using Faster-Whisper
        try:
            # print(f"Transcribing audio for {vid_dir}") # Giảm log thừa
            # Tham số beam_size=5 là mặc định, có thể điều chỉnh
            # language='vi' để ưu tiên tiếng Việt nếu biết trước
            segments, info = model.transcribe(audio_file, beam_size=5, language='vi', vad_filter=True)
            
            # Chuyển kết quả segments thành định dạng giống OpenAI Whisper
            result = {
                "text": " ".join([segment.text.strip() for segment in segments]),
                "segments": [
                    {
                        "id": i,
                        "seek": segment.seek,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "tokens": segment.tokens,
                        "temperature": segment.temperature,
                        "avg_logprob": segment.avg_logprob,
                        "compression_ratio": segment.compression_ratio,
                        "no_speech_prob": segment.no_speech_prob
                    } for i, segment in enumerate(segments)
                ],
                "language": info.language
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            # print(f"Saved transcription to {json_file}") # Giảm log thừa
        except Exception as e:
            print(f"Error transcribing audio for {vid_dir}: {str(e)}")


########## Video Captioning ##########
def run_vidcap(args):
    # Find all videos that have audio transcriptions
    videos_with_stt = []
    
    for video_dir in glob(os.path.join(args.root_dir, '*')):
        if not os.path.isdir(video_dir):
            continue
            
        video_files = glob(os.path.join(video_dir, '*.mp4'))
        audio_json = os.path.join(video_dir, 'audio.json')
        caption_file = os.path.join(video_dir, 'caption.txt')
        
        # Skip if no video files or already has caption
        if not video_files:
            print(f"No video file found in {video_dir}")
            continue
            
        if os.path.exists(caption_file):
            print(f"Caption already exists for {video_dir}, skipping")
            continue
            
        # Check if audio transcription exists and has content
        if os.path.exists(audio_json):
            try:
                with open(audio_json) as f:
                    audio_data = json.load(f)
                if audio_data['text']:
                    videos_with_stt.append(video_files[0])  # Use the first video file if multiple exist
            except Exception as e:
                print(f"Error reading audio json for {video_dir}: {str(e)}")
    
    if not videos_with_stt:
        print("No videos found with speech transcriptions for captioning")
        return
        
    print(f"Found {len(videos_with_stt)} videos for captioning")
    run_videos(args, videos_with_stt)


########## Gather ##########
def gather_info(args):
    # Find all video directories
    video_dirs = [d for d in glob(os.path.join(args.root_dir, '*')) if os.path.isdir(d)]
    
    if not video_dirs:
        print("No video directories found for gathering info")
        return
        
    print(f"Gathering information for {len(video_dirs)} video directories")
    
    for video_dir in tqdm(video_dirs, desc="Gathering info"):
        video_id = os.path.basename(video_dir)
        caption_path = os.path.join(video_dir, 'caption.txt')
        audio_json = os.path.join(video_dir, 'audio.json')
        info_json = os.path.join(video_dir, 'info.json')
        
        # Skip if info.json already exists
        if os.path.exists(info_json):
            print(f"Info already exists for {video_id}, skipping")
            continue
            
        # Skip if caption or audio doesn't exist
        if not os.path.exists(caption_path):
            print(f"Caption not found for {video_id}, skipping")
            continue
            
        if not os.path.exists(audio_json):
            print(f"Audio transcription not found for {video_id}, skipping")
            continue
        
        # Read audio transcription
        try:
            with open(audio_json) as f:
                audio_data = json.load(f)
        except Exception as e:
            print(f"Error reading audio json for {video_id}: {str(e)}")
            continue

        # Read caption
        try:
            with open(caption_path) as f:
                caption = f.read().strip()
        except Exception as e:
            print(f"Error reading caption for {video_id}: {str(e)}")
            continue

        # Save results
        try:
            results = {
                'audio': audio_data,
                'caption': caption
            }
            with open(info_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved info for {video_id}")
        except Exception as e:
            print(f"Error saving results for {video_id}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Extract information from videos in directory')
    
    # Thêm các tham số cần thiết
    parser.add_argument('--root_dir', type=str, default='D:/school/Thesis/ExFunTube/pipeline/test_videos',
                      help='Root folder containing videos')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of worker processes for parallel processing')
    parser.add_argument('--skip_audio', action='store_true',
                      help='Skip audio extraction step')
    parser.add_argument('--skip_whisper', action='store_true',
                      help='Skip speech-to-text step')
    parser.add_argument('--skip_caption', action='store_true',
                      help='Skip video captioning step')
    
    args = parser.parse_args()
    
    # Tạo thư mục gốc nếu chưa tồn tại
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir, exist_ok=True)
        print(f"Created root directory: {args.root_dir}")
    
    try:
        # Thực hiện các bước xử lý
        if not args.skip_audio:
            print('\n1. Extracting audio from videos...')
            extract_audios(args)
            print('✓ Audio extraction completed')
        
        if not args.skip_whisper:
            print('\n2. Running speech-to-text...')
            run_whisper(args)
            print('✓ Speech-to-text completed')
        
        if not args.skip_caption:
            print('\n3. Running video captioning...')
            run_vidcap(args)
            print('✓ Video captioning completed')
        
        print('\n4. Gathering all information...')
        gather_info(args)
        print('✓ Information gathering completed')
        
        print(f"\nAll processing completed successfully!")
        print(f"Results are stored in: {args.root_dir}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
