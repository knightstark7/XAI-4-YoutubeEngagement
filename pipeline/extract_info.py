'''
    Download video, extract audio, and perform speech-to-text and video captioning.
    Finally, gather all the results and store them as a json file.
'''

import argparse
import os
from glob import glob
import json
import torch
import yt_dlp
from multiprocessing import Pool
from pathlib import Path
import whisper
from zero_shot_video_to_text.run import run_videos
from tqdm import tqdm


########## Download Videos ##########
def filter(info, *, incomplete):
    '''Filter function for video download'''
    # Bỏ điều kiện kiểm tra duration
    return None

ydl_opts = {
        'match_filter' : filter,
        'quiet' : 'True',
        'noplaylist' : 'True',
        'format' : '136+140/137+140/136+m4a/137+m4a/mp4+140/18/22/mp4+m4a',
        'outtmpl' :  '%(id)s/%(title)s.mp4'
    }

def download(url):
    '''Given args, download the according video to the folder'''
    if 'channel' in url:
        return
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    try:
        error_code = ydl.download(url)
    except Exception as e:
        print(e)

def download_videos_(args):
    ydl_opts = {
        'match_filter' : filter,
        'quiet' : 'True',
        'noplaylist' : 'True',
        'format' : '136+140/137+140/136+m4a/137+m4a/mp4+140/18/22/mp4+m4a',
        'outtmpl' : '%(id)s/%(title)s.mp4'
    }
    urls = ['https://youtube.com/watch?v=' + i for i in torch.load(args.video_ids) if not os.path.exists(args.root_dir + f'/{i}')]
    with Pool(args.num_workers) as p:
        p.map(download, urls)

def download_videos(args):
    print("Starting video download process...")
    
    # Cập nhật ydl_opts với các tùy chọn tốt hơn
    ydl_opts = {
            'match_filter' : filter,
            'quiet' : 'True',
            'noplaylist' : 'True',
            'format' : '136+140/137+140/136+m4a/137+m4a/mp4+140/18/22/mp4+m4a',
            'outtmpl' :  '%(id)s/%(title)s.mp4'
        }
    
    try:
        # Load video IDs
        video_ids = torch.load(args.video_ids, weights_only=True)  # Sử dụng weights_only=True để tránh warning
        if isinstance(video_ids, torch.Tensor):
            video_ids = video_ids.tolist()
        elif isinstance(video_ids, list):
            video_ids = [str(v) for v in video_ids]
            
        print(f"Found {len(video_ids)} videos to process")
        
        # Tạo URLs
        urls = []
        for vid_id in video_ids:
            if not os.path.exists(os.path.join(args.root_dir, vid_id)):
                urls.append(f'https://youtube.com/watch?v={vid_id}')
                
        if not urls:
            print("No new videos to download. All videos already exist.")
            return
            
        print(f"Starting download of {len(urls)} videos...")
        
        # Tải từng video
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        for url in urls:
            try:
                print(f"\nDownloading: {url}")
                ydl.download([url])
                print(f"Successfully downloaded: {url}")
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
                continue
                
        print("\nVideo download process completed!")
        
    except Exception as e:
        print(f"Error in download_videos: {str(e)}")
        raise

########## Extract Audios ##########
def extract_audio(vid_dir):
    # extract audio from given video dir
    try:
        # Convert to Path object for better path handling
        video_path = Path(vid_dir)
        # Create audio directory if it doesn't exist
        audio_dir = video_path.parent / "audio"
        audio_dir.mkdir(exist_ok=True)
        # Set audio file path
        audio_path = audio_dir / "audio.wav"
        
        if not audio_path.exists():
            print(f"Extracting audio from: {video_path}")
            # Use absolute paths for ffmpeg command
            os.system(f'ffmpeg -y -i "{video_path.absolute()}" "{audio_path.absolute()}"')
            print(f"Audio extracted to: {audio_path}")
    except Exception as e:
        print(f'Error extracting audio from {vid_dir}: {str(e)}')

def extract_audios(args):
    # args.root_dir
    # |- {id}
    #    |- {title}.mp4
    #    |-  audio/
    #       |-  audio.wav
    videos = glob(os.path.join(args.root_dir, '*/*.mp4'))
    if len(videos) == 0:
        print('Warning: No videos found to process.')
        return
        
    print(f"Found {len(videos)} videos to extract audio from")
    with Pool(args.num_workers) as p:
        p.map(extract_audio, videos)


########## Speech-to-Text ##########
def run_whisper(args):
    videos = [i for i in glob(args.root_dir + '/*') if os.path.isdir(i)]
    # load model
    model = whisper.load_model("large-v2")
    for vid in videos:
        # transcribe audio and store result
        if os.path.exists(vid + '/audio.json'):
            continue
        result = model.transcribe(vid + '/audio.wav')
        with open(vid + '/audio.json', 'w') as f:
            json.dump(result, f, indent=2)
            

########## Video Captioning ##########
def run_vidcap(args):
    videos = glob(args.root_dir + '/*/*.mp4')
    audios = glob(args.root_dir + '/*/audio.json')
    videos_w_stt = []
    for vid, aud in zip(videos, audios):
        with open(aud) as f:
            aud_json = json.load(f)
        if len(aud_json['text']) > 0:
            videos_w_stt.append(vid)
    run_videos(args, videos_w_stt)


########## Gather ##########
def gather_info(args):
    try:
        # Load video IDs from .pt file
        video_list = torch.load(args.video_ids)
        if isinstance(video_list, torch.Tensor):
            video_list = video_list.tolist()
        elif isinstance(video_list, list):
            video_list = [str(v) for v in video_list]
    except Exception as e:
        print(f"Error loading video IDs: {str(e)}")
        return
    
    for video_id in tqdm(video_list):
        pth = os.path.join(args.root_dir, video_id, 'caption.txt')
        if not os.path.exists(pth):
            continue
            
        # Get base directory for the video
        video_dir = os.path.dirname(pth)
        
        # Read audio transcription
        audio_json = os.path.join(video_dir, 'audio.json')
        try:
            with open(audio_json) as f:
                audio_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find audio file for {video_id}")
            continue

        # Read caption
        try:
            with open(pth) as f:
                caption = f.read().strip()
        except:
            print(f"Warning: Could not read caption for {video_id}")
            continue

        # Save results
        output_file = os.path.join(video_dir, 'info.json')
        try:
            results = {
                'audio': audio_data,
                'caption': caption
            }
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving results for {video_id}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Extract information from YouTube videos')
    
    # Thêm các tham số cần thiết
    parser.add_argument('--root_dir', type=str, default='videos',
                      help='Root folder to store youtube videos')
    parser.add_argument('--video_ids', type=str, default='video_ids.pt',
                      help='Path to .pt file containing list of video IDs')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of worker processes for parallel processing')
    parser.add_argument('--skip_download', action='store_true',
                      help='Skip video download step')
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
        # Kiểm tra file video_ids.pt
        if not os.path.exists(args.video_ids):
            raise FileNotFoundError(f"Video IDs file not found: {args.video_ids}")
            
        # Load và kiểm tra video IDs
        video_ids = torch.load(args.video_ids)
        if not isinstance(video_ids, (list, torch.Tensor)):
            raise ValueError("Video IDs must be a list or tensor")
        print(f"Loaded {len(video_ids)} video IDs")
        
        # Thực hiện các bước xử lý
        if not args.skip_download:
            print('\n1. Downloading videos...')
            download_videos(args)
            print('✓ Video download completed')
        
        if not args.skip_audio:
            print('\n2. Extracting audio from videos...')
            extract_audios(args)
            print('✓ Audio extraction completed')
        
        if not args.skip_whisper:
            print('\n3. Running speech-to-text...')
            run_whisper(args)
            print('✓ Speech-to-text completed')
        
        if not args.skip_caption:
            print('\n4. Running video captioning...')
            run_vidcap(args)
            print('✓ Video captioning completed')
        
        print('\n5. Gathering all information...')
        gather_info(args)
        print('✓ Information gathering completed')
        
        print(f"\nAll processing completed successfully!")
        print(f"Results are stored in: {args.root_dir}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
